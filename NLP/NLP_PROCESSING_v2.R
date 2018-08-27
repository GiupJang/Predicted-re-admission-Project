#1.����ó�� -> /n /t, 2ĭ�������� ��� ��ĭ���� �ٲ���
#2.��ҹ��� ���� -> �Ǵ���а� ���� �� �۾�
#3.����ǥ�� -> ���ڵ��� ��� _number_�� �ٲ� �ν� or ���� �״�� �ν� or ���ڿ� ���� �ܾ ���� ���� ex) 3ȸ, 3�� �� ��� �� ������
#4.�����ȣ �� Ư������ ���� -> Ư�� ���̽��� �з��� ������ ����.  ���� �׷��ٸ� �Ǵ���а� �۾�, �ƴ϶�� ��� �� ������
#5.�ҿ�ܾ� ���� ->���� �����ؼ� ���� �ѱ��� �ʿ���� ǰ�� ���� / (����� ����� ����� �ٸ� ǰ��� �� ��� ó���Ұ����� ����) 
#6.��� ����ȭ ó�� -> �ѱ��� ����X,  ����� ��� �Ұ�����
#7.���׷�(n�� ���̾� �����ϴ� �ܾ���� ����) ->�ϳ��� ���п��� �ٿ����ϴ��� ���� �� ������ ���δٸ� �Ǵ���а� ������ �۾� #����� ������ ����ڷ�
#������ ex) Unique_word1 ���߿� POS_EXTRACTION�Լ� �����ϸ� ����ϱ� �״�� ���� �� �� list���� ã�Ƽ� �ٽ� �ٲ��� 
#8.���� ���λ�, ���̻� �����ؾ�����?  �´ٸ� �װ͸� �ϸ� �Ǵ��� # �ϴ� ������ �ؾ��� ���� ���ܿ� �ܾ� ������ �յڷ� ����??

#ubuntu ȯ�濡�� topicmodels install �� ctm.c:29:25: fatal error: gsl/gsl_rng.h: No such file or directory ���� ������
#sudo apt-get install libgsl0-dev �� �ذ�

#���� �غ����########################################
#XML_Parsing_Pro7���� file=""�� RDS ���� ��θ� ���ְ� ������ �� ���� �� ��.
#C:\Program Files\R\R-3.5.1\library\base\R\RProfile�� options(java.parameters = c("-Xmx16384m","-Dfile.encoding=UTF-8")) �߰� # KoNLP���� ���� 
#options(java.parameters = c("-Xmx16384m","-Dfile.encoding=UTF-8"))
#options("java.parameters")$java.parameters
######################################################

#POS ���� �Լ�
K_POS_EXTRACTION <- function(wordlist){
    #5.�ҿ� �ܾ� ����
    wordlist <- gsub('/F+','/CW+',wordlist)
    wordlist <- gsub('/NC+','/CW+',wordlist)
    
    pos_start <- as.vector(gregexpr('[^+]+\\/CW[+]',wordlist)[[1]]) # ����ǥ������ ���� �ɷ��� ��� 
    pos_length <- as.vector(attr(gregexpr('[^+]+\\/CW[+]',wordlist)[[1]],'match.length'))
    
    pos_end <- pos_start+pos_length-5
    
    word_data = rep(NA,length(pos_start))
    word <- c()
    for(i in 1:length(pos_start)){
        word_data[i] <- substr(wordlist,pos_start[i],pos_end[i])
        word <- paste(word,word_data[i])
    }
    word <- substr(word,2,nchar(word))
    
    return(word)
}

#������ ������ �κ��� ��ó�����ִ� �Լ�  
NLP_PROCESSING <- function(xmldf){
    #4.Ư�� ���� ���� �� ����
    xmldf <- gsub('&#x0D;', " ", xmldf) # ����� �������� ������ �ϳ��� ���� ���ִ� �κ��� ����. 
    xmldf <- gsub('&lt;', " ", xmldf)
    xmldf <- gsub('&gt;', " ", xmldf)
    xmldf <- gsub('&amp;', " ", xmldf)
    xmldf <- gsub('&quot;', " ", xmldf)
    
    xmldf <- gsub("[\\]","", xmldf)#Ư������ ����
    xmldf <- gsub("[\\+]|[\\{]|[\\}]|[\\(]|[\\)]|[\\<]|[\\>]"," ", xmldf)#Ư������ ����
    xmldf <- gsub("\\[","", xmldf)#Ư������ ����
    xmldf <- gsub("\\]","", xmldf)#Ư������ ����
    xmldf <- gsub("\\/","", xmldf)#Ư������ ����
    xmldf <- gsub("\\'"," ", xmldf)#Ư������ ����
    xmldf <- gsub('\\"'," ", xmldf)#Ư������ ����
    xmldf <- gsub("[~!@#$><%��=^&��*-:�ܡڢ�]"," ", xmldf)#Ư������ ����
    
    xmldf <-xmldf <- gsub(',', " ", xmldf) # �޸��� ��ĭ ����߷���.
    
    #2.��ҹ��� ����(���ð������� ���� ��)
    #xmldf<- toupper(xmldf) # �빮�� 
    xmldf<- tolower(xmldf)# �ҹ���
    
    xmldf <- gsub('[��-��]*','',xmldf)
    xmldf <- gsub('[��-��]*','',xmldf)
    
    #6.��� ����ȭ ó��
    #xmldf <- gsub(' are ',' be ',xmldf)
    #xmldf <- gsub(' are ',' be ',xmldf)
    #xmldf <- gsub(' is ',' be ',xmldf)
    
    #�Ǵ� �߿��������� �ܾ���� ���� �ʹ�
    #xmldf <- gsub('and|of|as|in',"",xmldf)# ��ó�� �� ���� ó�� �� ��.
    
    #7.���׷�
    #xmldf <- sub('[^A-Za-z ��-�R]*graphic[ _-]variant[^A-Za-z ��-�R]*','graphicvariant',xmldf) # KoNLP ó���� ������� �״�� ������ ������ �Ѵܾ�� �ٷ� ����.
    
    #�ѱ�, ��� �پ��ִ� ��쿡 ����߷���.
    pos_start <- as.vector(gregexpr('[^��-�R ]*[A-Za-z]+[^��-�R ]*',xmldf)[[1]]) # ����ǥ������ ���� �ɷ��� ��� 
    pos_length <- as.vector(attr(gregexpr('[^��-�R ]*[A-Za-z]+[^��-�R ]*',xmldf)[[1]],'match.length'))
    pos_end <- pos_start+pos_length-1
    
    word_data <- c()
    if(length(pos_start) > 0){  
        for(i in 1:length(pos_start)){
            word_data[i] <- substr(xmldf,pos_start[i],pos_end[i])
        }
        
        new_word_data <- paste("",toupper(word_data),"")
        
        for(i in 1:length(word_data)){
            xmldf <- sub(word_data[i],new_word_data[i],xmldf)
        }
    }
    xmldf<- tolower(xmldf)# �ٽ� �ҹ��� ó���� ����.
    
    
    #1.����ó��
    xmldf <- stringr::str_replace_all(xmldf,"[[:space:]]{1,}"," ")# ��ĭ�̻��� ���⸦ ��ĭ���� ����
    
    #�ٲ� ����
    xmldf <- paste(xmldf,'.',sep = '')#������ �ƴ� ��� ��¥, ��� �� �߿��� ������ �߸��� ��찡 ����. ex) 12-02-02 �ܾ� �ϳ������� 12-02-0 �� 2�� ����.
    #������ ������ . �߰����ָ� ���� �������� ����ǥ���Ŀ��� �Ÿ��� ������ �������Ŷ�� ������.
    
    return(xmldf)
}
#ǰ�� �м���
POS_ANALYSIS <- function(word_df){
    word_list <- KoNLP::SimplePos22(word_df)
    if(length(word_list) ==1){
        word_vector <- word_list[[1]]
        result_word_list <- c(word_vector)
    } 
    else{
        word_vector <- word_list[[1]]
        for (k in 2:length(word_list)){
            word_vector <- paste(word_vector,'+',word_list[[k]],sep = '')
        }
        result_word_list <- c(word_vector)
    }
    return(result_word_list)
}


# load packages
if(!require(rJava)) {
    install.packages('rJava')
}
if(!require(KoNLP)) {
    install.packages('KoNLP')
}
if(!require(devtools)) {
    install.packages('devtools')
}
#library(devtools)
#install_github('haven-jeon/NIADic/NIAdic', build_vignettes = TRUE)
if(!require(topicmodels)) {
    install.packages('topicmodels')
}
if(!require(openNLP)) {
    install.packages('openNLP')
}
if(!require(NLP)) {
    install.packages('NLP')
}
if(!require(parallel)) {
    install.packages("parallel")
}

Sys.setenv(JAVA_HOME="C:\\Program Files\\Java/jdk1.8.0_171") 
library(KoNLP)
library(rJava)
library(topicmodels)
library(stringr)
library(parallel)

# �ھ� ���� ȹ��
numCores <- parallel::detectCores() - 1
# Ŭ������ �ʱ�ȭ
myCluster <- parallel::makeCluster(numCores)

useSejongDic()
#mergeUserDic(dictionary_df)
#mergeUserDic(zz)
########MAIN CODE##############################################################



search_df <- result_xml_df[result_xml_df$`<MN>`=='������',] # �±�, �˻��� ���� ex) <MN>, '���'

tag ='<TD>' # NLP ó���ϰ� ���� tag �Է�

#�˻� �� tag�� NA�� ���� ����
search_df[,tag][is.na(search_df[,tag])] <- ""

for (i in nrow(search_df):1){# �ڿ������� ������ ���� �з��� ���� ���� �ʵ��� ó����.
    if(search_df[i,tag] == ""){
        search_df <- search_df[-i,]
    }
}

#NLP �� df �����
xml_df <- search_df[tag]

#NLP_PROCESSING �Լ��� ���� �ʱ� ����(����ó��)
word_df <- as.data.frame(parApply(myCluster,xml_df,1,NLP_PROCESSING))

#���¼� �м��� ��ġ��
result_word_list <- apply(word_df,1,POS_ANALYSIS)
#result_word_list <- parApply(myCluster,word_df,1,POS_ANALYSIS)
result_word_list<- unlist(result_word_list)

#���ϴ� ǰ�� ���� �� �ϳ��� �������� ������. (����ó��)
doc.list <- parallel::parLapply(myCluster,result_word_list,K_POS_EXTRACTION)

#DF �� ���� 
doc.tmp_df <- data.frame(unlist(doc.list),stringsAsFactors = FALSE)
doc.df <- data.frame(c(search_df['NOTE_ID'],doc.tmp_df,search_df['outcomeCount']),stringsAsFactors = FALSE)

#colname ����
colnames(doc.df) <- colnames(doc.df) <- c('NOTE_ID','NOTE_TEXT','outcomeCount')