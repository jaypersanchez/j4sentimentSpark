package com.kml.bigdata.sentiment;

import static com.mongodb.client.model.Filters.and;
import static com.mongodb.client.model.Filters.eq;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.bson.Document;

import com.mongodb.MongoClient;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoDatabase;

import scala.Tuple2;

public class TweetsSentiment {

  private static final String DATABASE_NAME = "social";
  private static final String REFERENCE_COLLECTION_NAME = "reference";
  private static final String TWITTER_COLLECTION_NAME = "twitter";

  private static MongoClient mongoClient;
  private static MongoDatabase database;

  public static void main(String[] args) {

    if (args.length < 5) {
      System.err.println("usage: TweetsSentiment context subject subject_caption inputPath runId");
      System.exit(2);
    }

    String context = args[0];
    String subject = args[1];
    String subjectCaption = args[2];
    String inputPath = args[3];
    String runId = args[4];

    initMongoClient();

    SparkConf conf = new SparkConf().setAppName("TweetSentiment");//.setMaster("local[*]");
    JavaSparkContext sc = new JavaSparkContext(conf);

    // final List<String> stopWords = sc.textFile("stopwords.txt").collect();

    final Map<String, Integer> sentimentWords = sc.textFile("/AFINNcsv.txt") //
        .map(l -> l.split(",")) //
        .filter(arr -> arr.length > 1) //
        .mapToPair(arr -> new Tuple2<String, Integer>(arr[0], Integer.valueOf(arr[1]))) //
        .collectAsMap();

    final Document refDoc = getReferenceDocForContext(context, runId);
    @SuppressWarnings("unchecked")
	final  List<String> topics = refDoc.get("topics", List.class);
    final String contextCaption=refDoc.getString("contextCaption");
    
    JavaRDD<String> tweets = sc.textFile(inputPath) //
        .filter(l -> !l.isEmpty() && l.contains(subject)) //
        .map(l -> l.split("\\^")) //
        .filter(arr -> (arr.length >= 8)) //
        .map(arr -> arr[7]) //
        .map(l -> l.toLowerCase());

    System.out.println(tweets.count());
    
    JavaPairRDD<String, Integer> scoresByTopic = tweets.flatMapToPair(tweet -> {
      List<Tuple2<String, String>> l = new ArrayList<Tuple2<String, String>>();
      for (String topic : topics) {
        if (tweet.contains(topic)) {
          l.add(new Tuple2<String, String>(topic, tweet));
        }
      }
      return l.iterator();
    }) //
        .mapValues(tweet -> Arrays.asList(tweet.split("\\s+"))) //
        .mapValues(list -> {
          Optional<Integer> tweetScore = list.stream()
              .map(word -> sentimentWords.getOrDefault(word, 0)).reduce((sc1, sc2) -> sc1 + sc2);
          return tweetScore.orElse(0);
        });

    scoresByTopic.cache();
    Map<String, Long> totalCounts = scoresByTopic.countByKey();
    Map<String, Long> positivesCounts = scoresByTopic.filter(t -> t._2 > 0).countByKey();
    Map<String, Long> negativesCounts = scoresByTopic.filter(t -> t._2 < 0).countByKey();

    List<Document> topicDocs = topics.stream()
        .map(topic -> new Document("topic", topic) //
            .append("totalNumberOfComments", totalCounts.get(topic)) //
            .append("totalPositiveComments", positivesCounts.get(topic)) //
            .append("totalNegativeComments", negativesCounts.get(topic)) //
        ).collect(Collectors.toList());

    boolean isNew = !hasPreviousDocument(context, subject);

    Document document = new Document("context", context)
    		.append("contextCaption", contextCaption)//
        .append("subject", subject)
        .append("subjectCaption", subjectCaption)//
        .append("topics", topicDocs);

    System.out.println(document);

    if (isNew)
      database.getCollection(TWITTER_COLLECTION_NAME).insertOne(document);
    else
      database.getCollection(TWITTER_COLLECTION_NAME).updateOne(
          and(eq("context", context), eq("subject", subject)), new Document("$set", document));

    sc.close();
    closeMongoClient();
  }

  private static boolean hasPreviousDocument(String context, String subject) {
    MongoCollection<Document> collection = database.getCollection(TWITTER_COLLECTION_NAME);
    Document doc = collection.find(and(eq("context", context), eq("subject", subject))).first();
    return doc != null;
  }

  @SuppressWarnings("unchecked")
  private static List<String> getReferenceForContext(String context, String runId) {

    MongoCollection<Document> collection = database.getCollection(REFERENCE_COLLECTION_NAME);
    Document runDoc = collection.find(eq("runId", runId)).first();
    List<String> ref = runDoc.get("topics", List.class);

    return ref;
  }
  
  @SuppressWarnings("unchecked")
  private static Document getReferenceDocForContext(String context, String runId) {
    MongoCollection<Document> collection = database.getCollection(REFERENCE_COLLECTION_NAME);
    Document runDoc = collection.find(eq("runId", runId)).first();
    return runDoc;
  }

  private static void initMongoClient() {
    mongoClient = new MongoClient();
    database = mongoClient.getDatabase(DATABASE_NAME);
  }

  private static void closeMongoClient() {
    mongoClient.close();
  }
}
