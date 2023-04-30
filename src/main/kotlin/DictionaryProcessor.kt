import java.io.File
import java.math.BigDecimal
import java.math.RoundingMode
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.roundToInt

private const val ENGLISH_REGEX = "[^A-Za-z ]"

private const val HTML_REGEX = "<(?:\"[^\"]*\"['\"]*|'[^']*'['\"]*|[^'\">])+>"

class DictionaryProcessor private constructor(
    private val path: String,
    private val percentTrainData: Int,
    private val dropData: Int,
    private val minLengthOfWord: Int,
    private val maxLengthOfWord: Int,
    private val limitBagOfWords: Int,
    private val probabilityScale: Int,
    private val testInclude: Boolean,
    private val stopWordProcessor: StopWordProcessor?,
    private val dropPmi: Boolean,
    private val pmiPercent: Int,
    private val dropIdf: Boolean,
    private val idfPercent: Int
){
    private val trainData: List<ReviewData>
    private val testData: List<ReviewData>
    private val setWordData: Set<WordData>
    private val probabilityOfPositive: Double
    private val probabilityOfNegative: Double
    private val countPositive: Int
    private val countNegative: Int
    private val lemmatizer: StanfordLemmatizer = StanfordLemmatizer()
    private val pmi: MutableMap<String, Int> = mutableMapOf()
    private val idf: MutableMap<String, BigDecimal> = mutableMapOf()

    init {
        val allData = getAllReviewData().shuffled()
        val size = allData.size
        trainData = allData.subList(0, size*percentTrainData / 100)
        testData = allData.subList(size*percentTrainData / 100, size)

        println("AllData size = $size")
        println("TrainData size = ${trainData.size}")
        println("TestData size =  ${testData.size}")

//        var bagOfWords = trainData.getBagOfLemmas()
        var bagOfWords = trainData.getBagOfWords()
        print("pmi = $pmi")
        trainData.idf(bagOfWords)
        print("idf = $idf")
//        if(dropPmi) bagOfWords = pmi.toList().sortedBy { (_, value) -> value }.toMap().keys
//            .take(pmi.size * pmiPercent / 100).toSet()
//
//                if(dropIdf) bagOfWords = idf.toList().sortedBy { (_, value) -> value }.toMap().keys
//            .take(idf.size * idfPercent / 100).toSet()
        println("First 10 words from bag of words: ")
        bagOfWords.forEach { println(it) }

        countPositive = trainData.getPositiveCount()
        countNegative = trainData.size - countPositive

        probabilityOfPositive = trainData.getPositiveProbability()
        probabilityOfNegative = (1 - probabilityOfPositive)
            .times(10.0.pow(probabilityScale).toInt())
            .roundToInt() / 10.0.pow(probabilityScale)
        println("Probability of positive reviews = $probabilityOfPositive")
        println("Probability of negative reviews = $probabilityOfNegative")

        val positiveTexts = trainData.getPositiveTexts()
        val negativeTexts = trainData.getNegativeTexts()

        setWordData = bagOfWords
            .asSequence()
            .mapIndexed { index, word ->
                if(index % 100 == 0) println("Train data pass $index elements")
                WordData(word = word,
                positiveProbability = BigDecimal(getPositiveCountOfWord(word, positiveTexts) /  countPositive.toDouble() ).setScale(probabilityScale, RoundingMode.FLOOR),
                negativeProbability = BigDecimal(getNegativeCountOfWord(word, negativeTexts) / countNegative.toDouble()).setScale(probabilityScale, RoundingMode.FLOOR)) }
            .toSet()

        setWordData.take(10).forEach { println("Word = ${it.word}, positive = ${it.positiveProbability}, negative = ${it.negativeProbability}") }

        if (testInclude) { test() }


    }

    fun getReviewScore(text: String) {
        val positiveProbability =
            setWordData.map {
                if(text.contains(it.word)) it.positiveProbability
                else BigDecimal(1.0).minus(it.positiveProbability)
            }.reduce { ans, num -> ans.multiply(num).setScale(100, RoundingMode.FLOOR)}
                .multiply(BigDecimal(probabilityOfPositive))

        val negativeProbability =  setWordData.map {
            if(text.contains(it.word)) it.negativeProbability
            else BigDecimal(1.0).minus(it.negativeProbability)
        }.reduce { ans, num -> ans.multiply(num) }
            .multiply(BigDecimal(probabilityOfNegative))

        println("Positive probability = $positiveProbability")
        println("Negative probability = $negativeProbability")
        println(if(positiveProbability >= negativeProbability) "This text is positive!" else "This text is negative!")
    }

    private fun test() {
        var countRightAnswers = 0
        var countRightPositiveAnswers = 0
        var countRightNegativeAnswers = 0
        var countWrongPositiveAnswers = 0
        var countWrongNegativeAnswers = 0

        var count = 0
        testData.forEach { data ->
            count++
            val positiveProbability =
                setWordData.asSequence().map {
                    if(data.review.contains(it.word)) it.positiveProbability
                    else BigDecimal(1.0).minus(it.positiveProbability)
                }.reduce { ans, num -> ans.multiply(num).setScale(100, RoundingMode.FLOOR)}
                    .multiply(BigDecimal(probabilityOfPositive))

            val negativeProbability =  setWordData.asSequence().map {
                if(data.review.contains(it.word)) it.negativeProbability
                else BigDecimal(1.0).minus(it.negativeProbability)
            }.reduce { ans, num -> ans.multiply(num) }
                .multiply(BigDecimal(probabilityOfNegative))

            if(positiveProbability >= negativeProbability && data.filmScore == 1) {
                countRightAnswers++
                countRightPositiveAnswers++
            }
            else if (positiveProbability < negativeProbability && data.filmScore == 0) {
                countRightAnswers++
                countRightNegativeAnswers++
            }
            else if (positiveProbability >= negativeProbability && data.filmScore == 0) countWrongPositiveAnswers++
            else if (positiveProbability < negativeProbability && data.filmScore == 1) countWrongNegativeAnswers++

            if(count == 100) println("Test data 100")
            if(count % 500 == 0) println("Test data $count")
        }

        println("Accuracy = ${BigDecimal(countRightAnswers.toDouble() / testData.size.toDouble()).setScale(probabilityScale, RoundingMode.FLOOR)}")

        println("Confusion matrix: ")
        println("TP = $countRightPositiveAnswers ")
        println("TN = $countRightNegativeAnswers ")
        println("FN = $countWrongNegativeAnswers ")
        println("FP = $countWrongPositiveAnswers ")

        val firstMetric = BigDecimal(countRightPositiveAnswers.toDouble() / (countRightPositiveAnswers.toDouble() + countWrongNegativeAnswers.toDouble())).setScale(probabilityScale, RoundingMode.FLOOR)
        val secondMetric = BigDecimal(countRightNegativeAnswers.toDouble() / (countRightNegativeAnswers.toDouble() + countWrongPositiveAnswers.toDouble())).setScale(probabilityScale, RoundingMode.FLOOR)
    println("Balanced Accuracy =" +
                " ${firstMetric.plus(secondMetric).div(BigDecimal(2.toDouble())).setScale(probabilityScale, RoundingMode.FLOOR)}")

        val precision = BigDecimal(countRightPositiveAnswers.toDouble() / (countRightPositiveAnswers.toDouble() + countWrongPositiveAnswers.toDouble())).setScale(probabilityScale, RoundingMode.FLOOR)
        val recall = BigDecimal(countRightPositiveAnswers.toDouble() / (countRightPositiveAnswers.toDouble() + countWrongNegativeAnswers.toDouble())).setScale(probabilityScale, RoundingMode.FLOOR)
        println("Precision = $precision")
        println("Recall = $recall")
        println("F-measure = ${precision.multiply(recall).multiply(BigDecimal(2.toDouble())).div (precision.plus(recall)).setScale(probabilityScale, RoundingMode.FLOOR)} ")
    }

    private fun List<ReviewData>.getBagOfWords() =
        map { it.review }
            .map(::processText)
            .flatMap { it.split(" ") }
            .filter { it.length in minLengthOfWord..maxLengthOfWord }
            .filterNot { stopWordProcessor?.isStopWord(it) ?: false }
            .onEach {
                val defaultVal = pmi.getOrDefault(it, 0)
                pmi[it] = defaultVal + 1
            }
            .toSet()
            .shuffled()
            .take(limitBagOfWords)
            .toSet()
    private fun List<ReviewData>.idf(bagOfWords: Set<String>) {
        bagOfWords.forEach { word ->
            idf[word] = BigDecimal(0.1)
        }
        this.map { it.review }
            .forEach{
                bagOfWords.forEach { word ->
                    if(it.contains(word)) {
                        idf[word] = idf[word]!!.plus(BigDecimal(1.0))
                    }
                }
            }

            idf.forEach { (k, v) ->
                idf[k] = BigDecimal(ln(BigDecimal(trainData.size).div(v).toDouble()))
            }
    }


    private fun List<ReviewData>.getBagOfLemmas() =
        asSequence().map { it.review }
            .map { it.replace("<br />", "") }
            .map(::processText)
            .mapIndexed{
                    index, text ->
                if(index % 100 == 0) println("Texts $index passed")
                lemmatizer.lemmatize(text)
            }
            .flatten()
            .flatMap { it.split(" ") }
            .filter { it.length in minLengthOfWord..maxLengthOfWord }
            .filterNot { stopWordProcessor?.isStopWord(it) ?: false }
            .onEach {
                val defaultVal = pmi.getOrDefault(it, 0)
                pmi[it] = defaultVal + 1
            }
            .toSet()
            .shuffled()
            .take(limitBagOfWords)
            .toSet()

    private fun List<ReviewData>.getPositiveProbability(): Double =
       getPositiveCount()
            .div(this.size.toDouble())
            .times(10.0.pow(probabilityScale).toInt())
            .roundToInt() / 10.0.pow(probabilityScale)

    private fun List<ReviewData>.getPositiveCount(): Int =
        map { it.filmScore }
            .filter { it == 1 }
            .count()

    private fun getAllReviewData(): List<ReviewData> =
        File(path).bufferedReader()
            .readLines()
            .drop(dropData)
            .map { ReviewData(it.split("\t")[1].toInt(), it.split("\t")[2]) }


    private fun List<ReviewData>.getPositiveTexts(): List<String> =
        filter { it.filmScore == 1 }
            .map { it.review }
            .map { it.replace("<br />", "") }
            .map (::processText)
            .flatMap { it.split(" ") }
            .toList()

    private fun List<ReviewData>.getNegativeTexts(): List<String> =
        filter { it.filmScore == 0 }
            .map { it.review }
            .map { it.replace("<br />", "") }
            .map (::processText)
            .flatMap { it.split(" ") }
            .toList()

    private fun getPositiveCountOfWord(word: String, positiveWords: List<String>): Int =
        positiveWords
            .filter { it == word }
            .count()

    private fun getNegativeCountOfWord(word: String, negativeWords: List<String>): Int =
        negativeWords
            .filter { it == word }
            .count()

    private fun processText(text: String): String = text.replace(ENGLISH_REGEX.toRegex(), "").replace(HTML_REGEX.toRegex(), "").lowercase()


    data class DictionaryBuilder(
        var path: String = "",
        var percentTrainData: Int = 80,
        var dropData: Int = 1,
        var minLengthOfWord: Int = 5,
        var maxLengthOfWord: Int = 10,
        var limitBagOfWords: Int = 10_000,
        var probabilityScale: Int = 5,
        var testInclude: Boolean = true,
        var stopWordProcessor: StopWordProcessor? = null,
        var dropPmi: Boolean = false,
        var pmiPercent: Int = 1,
        var dropIdf: Boolean = false,
        var idfPercent: Int = 1
    ){
        fun path(path: String) = apply { this.path = path }
        fun percentTrainData(percentTrainData: Int) = apply { this.percentTrainData = percentTrainData }
        fun dropData(dropData: Int) = apply { this.dropData = dropData }
        fun minLengthOfWord(minLengthOfWord: Int) = apply { this.minLengthOfWord = minLengthOfWord }
        fun maxLengthOfWord(maxLengthOfWord: Int) = apply { this.maxLengthOfWord = maxLengthOfWord }
        fun limitBagOfWords(limitBagOfWords: Int) = apply { this.limitBagOfWords = limitBagOfWords }
        fun probabilityScale(probabilityScale: Int) = apply { this.probabilityScale = probabilityScale }
        fun testInclude(testInclude: Boolean) = apply { this.testInclude = testInclude }
        fun stopWordProcessor(stopWordProcessor: StopWordProcessor) = apply { this.stopWordProcessor = stopWordProcessor }
        fun dropPmi(dropPmi: Boolean) = apply { this.dropPmi = dropPmi }
        fun pmiPercent(pmiPercent: Int) = apply { this.pmiPercent = pmiPercent }
        fun dropIdf(dropIdf: Boolean) = apply { this.dropIdf = dropIdf }
        fun idfPercent(idfPercent: Int) = apply { this.idfPercent = idfPercent }
        fun build() = DictionaryProcessor(
            path = path,
            percentTrainData = percentTrainData,
            dropData = dropData,
            minLengthOfWord = minLengthOfWord,
            maxLengthOfWord = maxLengthOfWord,
            limitBagOfWords = limitBagOfWords,
            probabilityScale = probabilityScale,
            testInclude = testInclude,
            stopWordProcessor = stopWordProcessor,
            dropPmi = dropPmi,
            pmiPercent = pmiPercent,
            dropIdf = dropIdf,
            idfPercent = idfPercent
        )
    }

}


