
 $(document).ready(function() {
  $("#predict").click(function() {
      $("#myModal").modal("show");
  });
});
class Tokenizer {
    /**
     * Class Tokenizer
     * @param filters a string where each element is a character that will be filtered from the texts. The default is all punctuation, plus tabs and line breaks, minus the ' character.
     * @param num_words the maximum number of words to keep, based on word frequency. Only the most common num_words-1 words will be kept.
     * @param oov_token if given, it will be added to word_index and used to replace out-of-vocabulary words during text_to_sequence calls
     * @param lower boolean. Whether to convert the texts to lowercase.
     * @return tokenizer object
     */
    constructor(config = {}) {
      this.filters = config.filters || /[\\.,/#!$%^&*;:{}=\-_`~()]/g;
      this.num_words = parseInt(config.num_words) || null;
      this.oov_token = config.oov_token || "";
      this.lower = typeof config.lower === "undefined" ? true : config.lower;
  
      // Primary indexing methods. Word to index and index to word.
      this.word_index = {};
      this.index_word = {};
  
      // Keeping track of word counts
      this.word_counts = {};
    }
  
    cleanText(text) {
      if (this.lower) text = text.toLowerCase();
      return text
        .replace(this.filters, "")
        .replace(/\s{2,}/g, " ")
        .split(" ");
    }
  
    /**
     * Updates internal vocabulary based on a list of texts.
     * @param texts	can be a list of strings, a generator of strings (for memory-efficiency), or a list of list of strings.
     */
  
    fitOnTexts(texts) {
      texts.forEach((text) => {
        text = this.cleanText(text);
        text.forEach((word) => {
          console.log("fitontext",this.word_counts[word])
          this.word_counts[word] = (this.word_counts[word] || 0) + 1;
        });
      });
  
      // Create words vector according to frequency (high to low)
      let vec = Object.entries(this.word_counts).sort((a, b) => b[1] - a[1]);
      // if oov_token is provided, add it to word_index / index_word
      if (this.oov_token) vec.unshift([this.oov_token, 0]);
      // Assign to word_index / index_word
      vec.every(([word, number], i) => {
        this.word_index[word] = i + 1;
        this.index_word[i + 1] = word;
        return true;
      });
    }
  
    /**
     * Transforms each text in texts to a sequence of integers.
     * Only top num_words-1 most frequent words will be taken into account. Only words known by the tokenizer will be taken into account.
     * @param texts	A list of texts (strings).
     * @return A list of sequences.
     */
    textsToSequences(texts) {
      // Only translate the top num_words(if provided) of words.
      return texts.map((text) =>
        this.cleanText(text).flatMap((word) =>
          this.word_index[word] &&
          (this.num_words === null || this.word_index[word] < this.num_words)
            ? this.word_index[word]
            : this.oov_token
            ? 1
            : []
        )
      );
    }
  
    /**
     * Returns a JSON string containing the tokenizer configuration. To load a tokenizer from a JSON string, use tokenizerFromJson(json_string).
     * @param replacer A function that transforms the results. (Passing to JSON.stringify())
     * @param space — Adds indentation, white space, and line break characters to thereturn-value JSON text to make it easier to read. (Passing to JSON.stringify())
     * @return A list of sequences.
     */
    toJson(replacer, space) {
      return JSON.stringify(
        {
          word_index: this.word_index,
          index_word: this.index_word,
          word_counts: this.word_counts,
        },
        replacer,
        space
      );
    }
  }
  
  /**
   * Create tokenizer from Json string
   * @param json_string JSON string encoding a tokenizer configuration.
   * @return A tokenizer object
   */
  function tokenizerFromJson(json_string) {
    
    const tokenizer = new Tokenizer();
//     const js = JSON.parse(json_string);
    // console.log("here in toke",js.config.word_counts["hotel"])

    tokenizer.word_index = JSON.parse(js.config.word_index);
    tokenizer.index_word = JSON.parse(js.config.index_word);
    tokenizer.word_counts = JSON.parse(js.config.word_counts);
    words=js.config.word_counts
    json=js
    tokenizer.num_words=JSON.parse(js.config.num_words);
    tokenizer.lower=JSON.parse(js.config.lower);
    tokenizer.oov_token=js.config.oov_token
    tokenizer.filters=js.config.filters

    return tokenizer;
  }
  
//   module.exports = { Tokenizer, tokenizerFromJson };
/* Loads trained model */
var vic=new Array()
var vocab
var words
var json

async function init() {
  
    model = undefined;
    
    console.log("start loading model")
    
    modellstm = await tf.loadLayersModel('./MODEL/model_lstm/model.json');
    modelgru = await tf.loadLayersModel('./MODEL/model_gru/model.json');
    
    console.log("model loaded..");
    const response = await fetch("./MODEL/tokenizer.json");
    const json = await response.json();
    vocab=JSON.parse(json)

      
  }
  function TokenisationAndPadding(text) {
    const tokenizer = tokenizerFromJson(vocab)
    console.log(tokenizer.word_counts['hotel'])
    tokenizer.fitOnTexts(text);
    let values = tokenizer.textsToSequences(text);    
    var valpadded = values.map(function(e) {
      const max_length = 300;
      const row_length = e.length 
      if (row_length > max_length){ // truncate
          return e.slice(row_length - max_length, row_length)
      }
      else if (row_length < max_length){ // pad
          return Array(max_length - row_length).fill(0).concat(e);
      }
      return e;
    });
    console.log(valpadded)
   
    return valpadded
  }

  init()
function PredictLstm(){
  var t = document.getElementById('Review').value;
  const text = [t];
  labels=['Negative','Positive']
  console.log(text)
  padded_sequence=TokenisationAndPadding(text)

  predictions = modellstm.predict(tf.tensor(padded_sequence));
  console.log("***********************")
  predictions=predictions.dataSync()
  console.log(predictions)
  console.log(Math.round(predictions))
  var res=labels[Math.round(predictions)]
  console.log(res)
  document.getElementById('Resultat').innerHTML = res;

  emoji="assets/img/"+res+".jpg"
  console.log(emoji)
  document.getElementById("emoji").src=emoji;


 }
 function PredictGru(){
  var t = document.getElementById('Review').value;
  const text = [t];
  labels=['Negative','Positive']
  console.log(text)
  padded_sequence=TokenisationAndPadding(text)

  predictions = modelgru.predict(tf.tensor(padded_sequence));
  console.log("***********************")
  predictions=predictions.dataSync()
  console.log(predictions)
  console.log(Math.round(predictions))
  var res=labels[Math.round(predictions)]
  console.log(res)
  document.getElementById('Resultat').innerHTML = res;

  emoji="assets/img/"+res+".jpg"
  console.log(emoji)
  document.getElementById("emoji").src=emoji;

 }

