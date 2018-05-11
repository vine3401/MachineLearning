from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName("miniproject").setMaster("local[*]")
sc = SparkContext().getOrCreate(conf=conf)
sc.setLogLevel("info")
intRDD = sc.parallelize([3,1,2,5,5])
print(intRDD.first())
