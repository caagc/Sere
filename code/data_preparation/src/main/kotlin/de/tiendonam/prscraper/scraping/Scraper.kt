package de.tiendonam.prscraper.scraping

import de.tiendonam.prscraper.database.*
import de.tiendonam.prscraper.utils.ExportRow
import de.tiendonam.prscraper.utils.ExportUtils
import de.tiendonam.prscraper.utils.RestClient
import de.tiendonam.prscraper.utils.parseJSON
import org.slf4j.LoggerFactory
import org.springframework.beans.factory.annotation.Value
import org.springframework.data.repository.findByIdOrNull
import org.springframework.stereotype.Service
import java.lang.RuntimeException
import java.time.OffsetDateTime
import javax.annotation.PostConstruct
import kotlin.system.exitProcess
import kotlinx.coroutines.*
import java.util.concurrent.ConcurrentLinkedQueue


@Service

class Scraper (
    private val restClient: RestClient,
    private val pullRequestRepo: PullRequestRepo,
    private val commitRepo: CommitRepo,
    private val commentRepo: CommentRepo,
    private val prBaseRepo: PRBaseRepo,
    private val prDiffRepo: PRDiffRepo,
    private val commentStreamService: CommentStreamService,
    private val scrapingStatusRepo: ScrapingStatusRepo,


    private val bufferQueue: ConcurrentLinkedQueue<Pair<PRBase, String>> = ConcurrentLinkedQueue(),
    private val bufferSize: Int = 30,
    @Value("\${scraping.repository}")
    private val repository: String,

    @Value("\${scraping.export.classified}")
    private val exportClassified: Boolean,

    @Value("\${scraping.export.all}")
    private val exportAll: Boolean
) {

    private val logger = LoggerFactory.getLogger(Scraper::class.java)

    @PostConstruct
    fun run() {
        logger.info("Starting...")

        val status = scrapingStatusRepo.findByIdOrNull(StatusKey.STAGE.name)
        var stage = if (status != null) StageValue.valueOf(status.value) else setStage(StageValue.PULL_REQUESTS)
        if (stage == StageValue.PULL_REQUESTS) {
            logger.info("Stage (1/4): pull requests")
            fetchPRs()
            stage = setStage(StageValue.COMMITS)
        }

        if (stage == StageValue.COMMITS) {
            logger.info("Stage (2/4): commits")
            fetchCommits()
            stage = setStage(StageValue.COMMENTS)
        }

        if (stage == StageValue.COMMENTS) {
            logger.info("Stage (3/4): comments")
            fetchComments()
            stage = setStage(StageValue.BASES)
        }

        if (stage == StageValue.BASES) {
            logger.info("Stage (4/4): fetch base files")
            fetchBases()
            stage = setStage(StageValue.DIFFS)
        }

        if (stage == StageValue.DIFFS) {
            logger.info("Stage (5/5): fetch diffs")
            fetchDiffs()
            setStage(StageValue.DONE)
        }

        if (exportClassified) {
            logger.info("Writing labeled results to csv...")

            // fetch rows
            val rows = commentRepo
                .findByClassTopicNotNull()
                .map { comment -> ExportRow(comment.id, comment.message, comment.classTopic?.name, comment.note, comment.author == comment.pullRequest.author) }

            ExportUtils.exportCSV(rows, "dataset_labeled.csv", preprocessing = false)
            ExportUtils.exportCSV(rows, "dataset_labeled_digested.csv", preprocessing = true)
            logger.info("Done.")
        }

        if (exportAll) {
            logger.info("Writing all results to csv...")

            // fetch rows (one by one)
            val rows = mutableListOf<ExportRow>()
            var counter = 0
            commentStreamService.doLineByLine { comment ->

                counter++
                if (counter % 10000 == 0) {
                    logger.info("Fetching all comments: $counter")
                }

                rows.add(ExportRow(comment.id, comment.message, comment.classTopic?.name, comment.note, comment.author == comment.pullRequest.author))
            }

            ExportUtils.exportCSV(rows, "dataset.csv", preprocessing = false)
            ExportUtils.exportCSV(rows, "dataset_digested.csv", preprocessing = true)
            logger.info("Done.")
        }
    }

    /**
     * fetch list of pull requests (meta data only)
     */
    fun fetchPRs(direction: String = "asc") {
        // fetch list of pull requests
        pullRequestRepo.deleteAll()
        var page = 1
        while (true) {
            val githubResponse = restClient
                .get("https://api.github.com/repos/$repository/pulls?state=all&sort=created&per_page=100&page=$page&direction=$direction")
                ?.parseJSON<List<PullRequestDto>>()
                ?: throw RuntimeException("Missing body in GET response. (fetching pulls)")

            if (githubResponse.isEmpty())
                break // reached last page

            val prParsed = githubResponse.map { dto ->
                PullRequest(
                    ghNumber = dto.number,
                    title = dto.title,
                    state = dto.state,
                    author = dto.user.login ?: "",
                    createdAt = OffsetDateTime.parse(dto.created_at)
                )
            }
            try {
                pullRequestRepo.saveAll(prParsed) // save to database
            } catch (e: Exception) {
                logger.error("Failed to save comments: ${e.message}")
            }
            logger.info("Fetched ${prParsed.size} PRs from page $page")
            page++
        }
    }

    /**
     * fetch comments for each pull request
     */
    fun fetchCommits() {
        commitRepo.deleteAll()
        val pullRequests = pullRequestRepo.findAll().toList()
        pullRequests.forEachIndexed { index, pullRequest ->
            val commitsDto = mutableListOf<CommitDto>()
            var page = 1
            while (true) {
                val githubResponse = restClient
                    .get("https://api.github.com/repos/$repository/pulls/${pullRequest.ghNumber}/commits?per_page=100&page=$page")
                    ?.parseJSON<List<CommitDto>>()
                    ?: throw RuntimeException("Missing body in GET response. (fetching commits)")

                if (githubResponse.isEmpty())
                    break // reached page after last page

                commitsDto.addAll(githubResponse)
                page++

                if (githubResponse.size < 100)
                    break // reached last page
            }
            
            val commitsParsed = commitsDto.map { dto ->
                Commit(
                    message = dto.commit.message,
                    pullRequest = pullRequest,
                    hash = dto.sha,
                    hashParent = dto.parents.firstOrNull()?.sha,
                    tree = dto.commit.tree.sha,
                    author = dto.author?.login ?: dto.commit.author.name,
                    createdAt = OffsetDateTime.parse(dto.commit.author.date)
                )
            }

            try {
                commitRepo.saveAll(commitsParsed)
            } catch (e: Exception) {
                logger.error("Failed to save comments: ${e.message}")
            }
            logger.info("Fetched ${commitsParsed.size} commits from PR #${pullRequest.ghNumber} (${index + 1} / ${pullRequests.size})")
        }
    }

    /**
     * fetch comments for each pull request
     */
    fun fetchComments() {
        // commentRepo.deleteAll()
        val fetchedPullRequestNumbers = commentRepo.findAll()
        .map { it.pullRequest.ghNumber }
        .toSet() // 使用 Set 以便高效查找
        // logger.info("Fetched ${fetchedPullRequestNumbers.size} pull requests from comments")
        val pullRequests = pullRequestRepo.findAll().toList()
        pullRequests.forEachIndexed { index, pullRequest ->
            if (fetchedPullRequestNumbers.contains(pullRequest.ghNumber)) {
                // logger.info("Skip PR #${pullRequest.ghNumber} (${index + 1} / ${pullRequests.size}) - Already fetched")
                return@forEachIndexed // 跳过已经爬取过的 pull request
            }
            val commentsDtoA = mutableListOf<CommentDto>()
            val commentsDtoB = mutableListOf<CommentDto>()
            var page = 1
            while (true) {
                val githubResponse = restClient
                    .get("https://api.github.com/repos/$repository/pulls/${pullRequest.ghNumber}/comments?per_page=100&page=$page")
                    ?.parseJSON<List<CommentDto>>()

                if (githubResponse == null) {
                    logger.error("Failed to fetch comments for PR #${pullRequest.ghNumber} via review api")
                    page++
                    continue
                }

                if (githubResponse.isEmpty())
                    break // reached page after last page

                commentsDtoA.addAll(githubResponse)
                page++

                if (githubResponse.size < 100)
                    break // reached last page
            }


            page = 1
            while (true) {
                val githubResponse = restClient
                    .get("https://api.github.com/repos/$repository/issues/${pullRequest.ghNumber}/comments?per_page=100&page=$page")
                    ?.parseJSON<List<CommentDto>>()
                
                if (githubResponse == null) {
                    logger.error("Failed to fetch comments for PR #${pullRequest.ghNumber} via issue api")
                    page++
                    continue
                }

                if (githubResponse.isEmpty())
                    break // reached page after last page

                commentsDtoB.addAll(githubResponse)
                page++

                if (githubResponse.size < 100)
                    break // reached last page
            }

            val commentsDto = commentsDtoA + commentsDtoB
            val commentsParsed = commentsDto.map { dto ->
                val commit = dto.original_commit_id?.let {
                    commitRepo.findByHashAndPullRequest(it, pullRequest).firstOrNull()
                }
                val commitFallback = dto.commit_id?.let {
                    commitRepo.findByHashAndPullRequest(it, pullRequest).firstOrNull()
                }
    
                Comment(
                    message = dto.body,
                    ghId = dto.id,
                    ghReplyId = dto.in_reply_to_id,
                    pullRequest = pullRequest,
                    commit = commit,
                    commitFallback = commitFallback,
                    hunkDiff = dto.diff_hunk,
                    hunkFile = dto.path,
                    author = dto.user?.login ?: "",
                    createdAt = OffsetDateTime.parse(dto.created_at),
                    original_position = dto.original_position,
                    position = dto.position,
                )
            }.sortedBy { comment -> comment.createdAt }
    
            try {
                commentRepo.saveAll(commentsParsed)
            } catch (e: Exception) {
                logger.error("Failed to save comments: ${e.message}")
            }
            // logger.info("Fetched ${commentsParsed.size} comments from PR #${pullRequest.ghNumber} (${index + 1} / ${pullRequests.size})")
        }
    }

    private suspend fun fetchFileContent(path: String, baseCommitSha: String): String? {
        val urls = listOf(
            "https://cdn.jsdelivr.net/gh/$repository@$baseCommitSha/$path",
            "https://raw.githubusercontent.com/$repository/$baseCommitSha/$path",
            "https://raw.gitmirror.com/$repository/$baseCommitSha/$path"
        )

        var fileContent: String? = null
        for (url in urls) {
            fileContent = restClient.get_no_token(url)
            if (fileContent != null) {
                break
            }
        }

        if (fileContent == null) {
            logger.error("Failed to fetch base file $path for commit $baseCommitSha")
        }

        return fileContent
    }

    private fun processBuffer() {
        runBlocking {
            val deferredResults = mutableListOf<Deferred<PRBase?>>()

            // 并发处理获取文件内容
            while (bufferQueue.isNotEmpty()) {
                val (prBase, baseCommitSha) = bufferQueue.poll()
                deferredResults.add(async(Dispatchers.IO) {
                    val fileContent = fetchFileContent(prBase.filePath, baseCommitSha)
                    // 仅在 fileContent 不为 null 时返回一个 PRBase 实例
                    if (fileContent != null) {
                        PRBase(
                            pullRequest = prBase.pullRequest,
                            filePath = prBase.filePath,
                            hash = prBase.hash,
                            fileContent = fileContent
                        )
                    } else {
                        null // 返回 null 表示未成功获取文件内容
                    }
                })
            }

            // 等待所有网络请求结束
            val results = deferredResults.awaitAll()

            // 按顺序保存到数据库
            results.filterNotNull()
            .forEach { prBase ->
                try {
                    prBaseRepo.save(prBase)
                }
                catch (e: Exception) {
                    logger.error("Failed to save base file ${prBase.filePath} for PR #${prBase.pullRequest.ghNumber}")
                    return@forEach
                }
                // 更新缓存，标记已爬取
                logger.info("Fetched base file ${prBase.filePath} for PR #${prBase.pullRequest.ghNumber}")
            }
        }
    }

    fun fetchBase(existingBases: MutableSet<Pair<String, String>>, path: String, baseCommitSha: String, pullRequest: PullRequest, commit_flag: String) {
        if (path.isEmpty()) {
            return
        }
        if (existingBases.contains(Pair(path, commit_flag))) {
            logger.info("Skipped already fetched base file $path for PR #${pullRequest.ghNumber}")
            return
        }
        val prBase = PRBase(
            pullRequest = pullRequest,
            filePath = path,
            hash = commit_flag,
            fileContent = ""
        )
        bufferQueue.add(Pair(prBase, baseCommitSha))
        if (bufferQueue.size >= bufferSize) {
            processBuffer()
        }
    }

    fun fetchBases() {
        // 加载已爬取的基础文件 (filePath 和 hash 组合)
        val existingBases = prBaseRepo.findAll()
            .map { Pair(it.filePath, it.hash) }
            .toMutableSet()  // 转换为集合以方便检查
    
        val comments = commentRepo.findAll()
            .filter { it.hunkDiff != null && it.hunkFile != null }
        val baseCommits = mutableMapOf<Long, String>() // pr_id -> base_commit_sha
    
        comments.forEach { comment ->
            // 从缓存中获取 PR 的基础提交 SHA
            val pullRequest = comment.pullRequest
            val baseCommitSha = baseCommits[pullRequest.id] ?: run {
                try {
                    val prResponse = restClient
                        .get("https://api.github.com/repos/$repository/pulls/${pullRequest.ghNumber}")
                        ?.parseJSON<Map<String, Any>>()
                        ?: return@forEach // 跳过当前循环
                    
                    prResponse["base"]?.let { (it as Map<String, String>)["sha"] } ?: null
                } catch (e: Exception) {
                    // 捕获异常，返回 null
                    null
                }
            }
    
            // 如果 baseCommitSha 为 null，则跳过当前循环
            if (baseCommitSha == null) {
                return@forEach
            }
            
            baseCommits[pullRequest.id] = baseCommitSha
            val path = comment.hunkFile ?: ""
    
            fetchBase(existingBases, path, baseCommitSha, pullRequest, "hash: $baseCommitSha")
            existingBases.add(Pair(path, "hash: $baseCommitSha"))
        }
        processBuffer()
    }
    
    
    
    fun fetchDiffs() {
        // 加载已爬取的 diff 数据 (filePath 和 commit.hash 组合)
        val existingDiffs = prDiffRepo.findAll()
            .map { Pair(it.filePath, it.commit.id) }
            .toSet()  // 转换为集合以方便检查
        val existingBases = prBaseRepo.findAll()
            .map { Pair(it.filePath, it.hash) }
            .toMutableSet()  // 转换为集合以方便检查
        // 获取所有 pr_base 数据，并按 prId 分类
        val prBasesByPrId = prBaseRepo.findAll().groupBy { it.pullRequest.id }
    
        prBasesByPrId.forEach { (prId, prBases) ->
            val commits = commitRepo.findByPullRequestId(prId).sortedBy { it.createdAt }
    
            val filePaths = prBases.map { it.filePath }.toSet()
    
            commits.forEach { commit ->
                val commitResponse = restClient
                    .get("https://api.github.com/repos/$repository/commits/${commit.hash}")
                    ?.parseJSON<Map<String, Any>>()
    
                val files = (commitResponse?.get("files") as? List<*>)?.filterIsInstance<Map<String, Any>>()
                    ?: return@forEach

                val pullRequest = prBases.first().pullRequest
                files.forEach { file ->
                    val filePath = file["filename"] as? String
                    val fileDiff = file["patch"] as? String
    
                    // 检查是否已经爬取相同的 filePath 和 commit.hash 组合
                    if (filePath != null && filePaths.contains(filePath) && fileDiff != null) {
                        val fileCommitPair = Pair(filePath, commit.id)
                        if (existingDiffs.contains(fileCommitPair)) {
                            logger.info("Skipped already fetched diff for $filePath in PR #${pullRequest.ghNumber}")
                            return@forEach // 跳过此文件
                        }
    
                        val prDiff = PRDiff(
                            pullRequest = pullRequest,
                            commit = commit,
                            filePath = filePath,
                            fileDiff = fileDiff
                        )
                        try {
                            prDiffRepo.save(prDiff)
                        }
                        catch (e: Exception) {
                            logger.error("Failed to save diff for $filePath in PR #${pullRequest.ghNumber}")
                            return@forEach
                        }
    
                        logger.info("Fetched diff for $filePath in PR #${pullRequest.ghNumber}")
                    }
                }

                val parents = (commitResponse?.get("parents") as? List<*>)?.filterIsInstance<Map<String, Any>>()
                if (parents != null && parents.size > 1) {
                    files.forEach { file ->
                        val filePath = file["filename"] as? String
                        fetchBase(existingBases, filePath ?: "", commit.hash, pullRequest, "id: ${commit.id}")
                        existingBases.add(Pair(filePath ?: "", "id: ${commit.id}"))
                    }
                }
            }
        }
        processBuffer()
    }
    
    

    /**
     * save stage to database and returns its value
     */
    private fun setStage(stage: StageValue): StageValue {
        scrapingStatusRepo.save(ScrapingStatus(StatusKey.STAGE.name, stage.name))
        return stage
    }

    private fun extractModifiedLines(diff: String): List<String> {
        return diff.split("\n")
            .filter { it.startsWith("+") || it.startsWith("-") }
            .map { it.substring(1).trim() } // Remove the '+' or '-' and trim whitespace
    }
    
    private fun splitPatch(patch: String): List<String> {
        return patch.split(Regex("(?<=\\n)(?=@@)")) // 在@@前面加上换行符进行分割
    }
}