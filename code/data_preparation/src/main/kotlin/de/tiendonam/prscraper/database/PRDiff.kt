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
data class PRDiff (
    @Id @GeneratedValue(strategy = GenerationType.SEQUENCE)
    var id: Long = 0,

    @ManyToOne(optional = false) @OnDelete(action = OnDeleteAction.CASCADE)
    var pullRequest: PullRequest = PullRequest(),

    @ManyToOne(optional = false) @OnDelete(action = OnDeleteAction.CASCADE)
    var commit: Commit = Commit(),

    @Column(nullable = false, columnDefinition = "TEXT")
    var filePath: String = "",

    @Column(nullable = false, columnDefinition = "TEXT")
    var fileDiff: String = ""
)

@Repository
interface PRDiffRepo : CrudRepository<PRDiff, Long> {
    fun findByPullRequest(pullRequest: PullRequest): List<PRDiff>
}
