package edu.elte.spring.loris.spark.model

import org.apache.spark._
import org.apache.spark.rdd._
import com.typesafe.config._
import scala.math.random
import it.nerdammer.spark.hbase._

import org.apache.hadoop.hbase.HBaseConfiguration
import org.apache.hadoop.hbase.HColumnDescriptor
import org.apache.hadoop.hbase.HTableDescriptor
import org.apache.hadoop.hbase.client._
import org.apache.hadoop.hbase.TableName
import org.apache.hadoop.conf.Configuration

import org.apache.hadoop.hbase.filter._
import org.apache.hadoop.hbase.filter.SingleColumnValueFilter
import org.apache.hadoop.hbase.filter.FamilyFilter
import org.apache.hadoop.hbase.filter.BinaryComparator
import org.apache.hadoop.hbase.util.Bytes

object NewFeeds {

  def getFreshFeeds(maxSize: Int): List[FeedEntry] = {
    //Csatlakozás HBase-re
    val config: Configuration = HBaseConfiguration.create()
    val connection: Connection = ConnectionFactory.createConnection(config)
    val admin: Admin = connection.getAdmin()

    val feedList = scala.collection.mutable.ListBuffer.empty[FeedEntry]

    //Lekérdezés összeállítása
    val table: Table = connection.getTable(TableName.valueOf("loris:FeedEntry"))
    val s = new Scan()
    s.addColumn(Bytes.toBytes("FeedEntry"), Bytes.toBytes("CONTENT"))
    s.addColumn(Bytes.toBytes("FeedEntry"), Bytes.toBytes("CHANNEL"))
    s.addColumn(Bytes.toBytes("FeedEntry"), Bytes.toBytes("TITLE"))
    s.addColumn(Bytes.toBytes("FeedEntry"), Bytes.toBytes("LABELED"))  
    val filter: Filter = new SingleColumnValueFilter(Bytes.toBytes("FeedEntry"), Bytes.toBytes("LABELED"), CompareFilter.CompareOp.EQUAL,
      new BinaryComparator(Bytes.toBytes(false)))
    s.setFilter(filter)
    val scanner: ResultScanner = table.getScanner(s)

    //Adatok lekérése
    val sIter = scanner.iterator
    var act : Int = 0
    while (sIter.hasNext && act < maxSize) {
      val current = sIter.next
      feedList.append(new FeedEntry(
        Bytes.toString(current.getRow), 
        Bytes.toString(current.getValue(Bytes.toBytes("FeedEntry"), Bytes.toBytes("TITLE"))),
        Bytes.toString(current.getValue(Bytes.toBytes("FeedEntry"), Bytes.toBytes("CONTENT"))),
        Bytes.toString(current.getValue(Bytes.toBytes("FeedEntry"), Bytes.toBytes("CHANNEL"))),
        Bytes.toBoolean(current.getValue(Bytes.toBytes("FeedEntry"), Bytes.toBytes("LABELED")))))
        act = act + 1
    }
    
    scanner.close()
    connection.close()
    
    feedList.toList
  }
}