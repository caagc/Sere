package de.tiendonam.prscraper.database

import org.hibernate.annotations.OnDelete
import org.hibernate.annotations.OnDeleteAction
import org.springframework.data.repository.CrudRepository
import org.springframework.stereotype.Repository
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.OffsetDateTime
import java.util.stream.Stream
import javax.persistence.*

@Entity
data class PRBase (
    @Id @GeneratedValue(strategy = GenerationType.SEQUENCE)
    var id: Long = 0,

    @ManyToOne(optional = false) @OnDelete(action = OnDeleteAction.CASCADE)
    var pullRequest: PullRequest = PullRequest(),

    @Column(nullable = false)
    var hash: String = "",

    @Column(nullable = false, columnDefinition = "TEXT")
    var filePath: String = "",

    @Column(nullable = false, columnDefinition = "TEXT")
    var fileContent: String = ""
)

@Repository
interface PRBaseRepo : CrudRepository<PRBase, Long> {
    fun findByPullRequest(pullRequest: PullRequest): List<PRBase>
    fun findByPullRequestAndFilePath(pullRequest: PullRequest, filePath: String): List<PRBase>
}
