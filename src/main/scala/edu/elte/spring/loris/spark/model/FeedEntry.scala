package edu.elte.spring.loris.spark.model

case class FeedEntry(rowid: String, title: String, content: String, channel: String, labeled: Boolean) extends Serializable
 