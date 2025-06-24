# Explanations about the docker image: https://jetblue-jupyter.blue.ygrid.yahoo.com:9999/nb/notebooks/projects/notebooks/jupyter/samples[…]/Jupyter_Demo_2.0__Custom_Python_Libraries.ipynb

# ml image docker doc: https://git.ouryahoo.com/hadoop-user-images/ml/blob/master/image-deploy.yaml
# https://tools.ygrid.yahoo.com:4443/abft/#/status/huserimages

# Ariel's project  https://git.ouryahoo.com/haifa-labs-mail/Geonosis/tree/dev

# queue gpu_v100   or   default
export QUEUE=gpu_v100
export USER=<username>
export BASE_PATH=hdfs://jetblue-nn1.blue.ygrid.yahoo.com:8020/projects/moneyball/${USER}/low_rank_experiments
export SRC_PATH=${BASE_PATH}/src
# ------------------------------------- #

${SPARK_HOME}/bin/spark-submit \
--master yarn \
--deploy-mode cluster \
--queue ${QUEUE} \
--executor-memory 20G \
--driver-memory 20G \
--conf spark.executor.resource.gpu.amount=1 \
--conf spark.oath.dockerImage=ml/rhel8_mlbundle:2021.12.1 \
--conf spark.driver.memoryOverhead=10G \
--conf spark.executor.memoryOverhead=6G \
--conf spark.hadoop.hive.metastore.uris=thrift://bassniumblue-hcat.blue.ygrid.yahoo.com:50513 \
--conf spark.kerberos.access.hadoopFileSystems=hdfs://bassniumblue-nn1.blue.ygrid.yahoo.com:8020 \
--conf spark.yarn.maxAppAttempts=1 \
--conf spark.ui.view.acls=* \
--archives ${SRC_PATH}/torchmetrics.zip \
--py-files ${SRC_PATH}/src.zip \
${SRC_PATH}/run_pyspark_experiment.py

#
#  ml/rhel8_mlbundle:2023.03.9

#--conf spark.pyspark.python=/opt/vz/1.0/bin/python \
#--conf spark.pyspark.driver.python=/opt/vz/1.0/bin/python \
