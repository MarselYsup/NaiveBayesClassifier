import java.util.*

fun main(args: Array<String>) {
   val path = "src/main/resources/labeledTrainData.tsv"
   val stopPath = "src/main/resources/stopwords.txt"
   val dictionaryProcessor = DictionaryProcessor.DictionaryBuilder()
       .path(path)
       .percentTrainData(80)
       .limitBagOfWords(1000)
       .stopWordProcessor(StopWordProcessor(stopPath))
//       .dropPmi(true)
//       .pmiPercent(60)
       .dropData(1)
        .build()
    while (true) {
        val text = Scanner(System.`in`).nextLine()
        dictionaryProcessor.getReviewScore(text)
    }





}//сделал в виде билдера