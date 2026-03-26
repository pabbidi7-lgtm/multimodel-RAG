sudo docker compose up -d
[sudo] password for clouduser01: 
WARN[0000] /home/clouduser01/jaswanth/docker-compose.yml: the attribute `version` is obsolete, it will be ignored, please remove it to avoid potential confusion 
[+] up 2/3
 ! Image redis:7                                        Interrupted                                       1.8s 
 ⠇ Image milvusdb/milvus:v2.4.4                         Pulling                                           1.8s 
 ✘ Image nvcr.io/nvidia/nemo-retriever/nv-ingest:25.9.0 Error error from registry: Access Denied          1.8s 
Error response from daemon: error from registry: Access Denied
(myenv) clouduser01@AZRCIDEVNIVIDIA:~/jaswanth$ sudo docker login nvcr.io
Authenticating with existing credentials... [Username: $oauthtoken]

i Info → To login with a different account, run 'docker logout' followed by 'docker login'


Login Succeeded
(myenv) clouduser01@AZRCIDEVNIVIDIA:~/jaswanth$ 
