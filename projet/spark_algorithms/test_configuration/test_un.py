"""SimpleApp.py"""
from pyspark import SparkContext

if __name__ == '__main__':
    logFile = "../../../data/test_spark.txt"  # Should be some file on your system
    sc = SparkContext("local", "Simple App")
    logData = sc.textFile(logFile).cache()

    numAs = logData.filter(lambda s: 'a' in s).count()
    numBs = logData.filter(lambda s: 'b' in s).count()

    print("Lines with a: %i, lines with b: %i" % (numAs, numBs))
