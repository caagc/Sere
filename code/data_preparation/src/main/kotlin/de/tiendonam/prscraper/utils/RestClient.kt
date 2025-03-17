package de.tiendonam.prscraper.utils

import org.slf4j.LoggerFactory
import org.springframework.beans.factory.annotation.Value
import org.springframework.http.HttpEntity
import org.springframework.http.HttpHeaders
import org.springframework.http.HttpMethod
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import org.springframework.web.client.RestTemplate
import java.time.Instant
import org.springframework.web.client.HttpClientErrorException
import org.springframework.web.client.HttpServerErrorException

import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

@Service
class RestClient(
    @Value("\${scraping.auth}")
    private val authorizationTokens: String
) {

    private val logger = LoggerFactory.getLogger(RestClient::class.java)
    private val rest = RestTemplate()
    private var tokens = authorizationTokens.split(",").map { it.trim() }.toMutableList()
    private var currentTokenIndex = 0
    private var consecutiveSwitchCount = 0
    private var headers = createHeaders(tokens[currentTokenIndex])

    private fun createHeaders(token: String): HttpHeaders {
        return HttpHeaders().apply {
            add("Content-Type", "application/json")
            add("Accept", "*/*")
            add("Authorization", "token $token")
        }
    }

    fun get(url: String): String? {
        var retryCount = 0
        val maxRetryCount = 3
        while (true) {
            try {
                val requestEntity = HttpEntity("", headers)
                logger.info("GET $url")
                val responseEntity = rest.exchange(url, HttpMethod.GET, requestEntity, String::class.java)
                logger.info("Response: ${responseEntity.statusCode}")
                val remaining = responseEntity.headers["X-RateLimit-Remaining"]?.firstOrNull()
                val resetAt = responseEntity.headers["X-RateLimit-Reset"]?.firstOrNull()?.toLongOrNull()

                if (remaining != null) {
                    logger.info("X-RateLimit-Remaining: $remaining")
                }
                if (resetAt != null) {
                    val remainingTime = resetAt - Instant.now().epochSecond
                    val min = remainingTime / 60
                    val sec = remainingTime % 60
                    val minString = min.toString().padStart(2, '0')
                    val secString = sec.toString().padStart(2, '0')
                    logger.info("X-RateLimit-Reset: in $minString:$secString")
                }
                consecutiveSwitchCount = 0 // Reset the switch count after a successful request
                return responseEntity.body
            } catch (e: HttpClientErrorException.Forbidden) {
                logger.warn("API rate limit exceeded for the current token.")
                if (!switchToken()) {
                    logger.error("No more tokens available.")
                    Thread.sleep(10 * 60 * 1000) // Sleep for 10 minutes before retrying
                }
                logger.info("Switched to a new token. Retrying...")
                Thread.sleep(60 * 1000) // Sleep for 1 minute before retrying
            } catch (e: Exception) {
                logger.error("Error fetching URL: $url, ${e.message}")
                //if "Bad credentials" in e.message, exit
                if (e.message?.contains("Bad credentials") == true) {
                    logger.error("Bad credentials detected, exiting...")
                    System.exit(1)
                }
                return null
            }
        }
    }

    fun get_no_token(url: String): String? {
        var connection: HttpURLConnection? = null
        try {
            val urlObj = URL(url)
            connection = urlObj.openConnection() as HttpURLConnection
            connection.requestMethod = "GET"
            connection.setRequestProperty("Accept", "*/*")
            connection.connectTimeout = 20000
            connection.readTimeout = 20000
    
            val responseCode = connection.responseCode
            if (responseCode == HttpURLConnection.HTTP_OK) {
                connection.inputStream.bufferedReader().use { reader ->
                    return reader.readText()
                }
            } else {
                logger.warn("Failed to fetch URL: $url, Response Code: $responseCode")
                return null
            }
        } catch (e: IOException) {
            logger.error("Error fetching URL: $url, ${e.message}")
            return null
        } finally {
            connection?.disconnect()
        }
    }    

    private fun switchToken(): Boolean {
        currentTokenIndex = (currentTokenIndex + 1) % tokens.size
        consecutiveSwitchCount++
        headers = createHeaders(tokens[currentTokenIndex])
        if (consecutiveSwitchCount >= tokens.size) {
            return false // All tokens exhausted
        }
        return true
    }

    fun post(url: String, json: String = ""): String? {
        val requestEntity = HttpEntity(json, headers)
        val responseEntity = rest.exchange(url, HttpMethod.POST, requestEntity, String::class.java)
        return responseEntity.body
    }

    fun put(url: String, json: String = ""): HttpStatus {
        val requestEntity = HttpEntity(json, headers)
        val responseEntity = rest.exchange(url, HttpMethod.PUT, requestEntity, String::class.java)
        return responseEntity.statusCode
    }

    fun delete(url: String): HttpStatus {
        val requestEntity = HttpEntity("", headers)
        val responseEntity = rest.exchange(url, HttpMethod.DELETE, requestEntity, String::class.java)
        return responseEntity.statusCode
    }
}
