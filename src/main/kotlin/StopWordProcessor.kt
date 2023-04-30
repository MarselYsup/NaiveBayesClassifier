import java.io.File

class StopWordProcessor(path: String) {
    private val stopWordSet = mutableSetOf<String>()
    init {
        File(path).bufferedReader()
            .readLines()
            .forEach { stopWordSet.add(it) }

    }

    fun isStopWord(word: String): Boolean = stopWordSet.contains(word)
}