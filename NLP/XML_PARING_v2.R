#���� �غ����########################################
#PatientLevelPrediction.R ���� �� ������ ��.
######################################################
library(progress)

#XML_parser -> ����ǥ������ ���� �Ľ� ����

XML_PARSING <- function(xmlList){
    pattern_start <- as.vector(gregexpr('<[^/<>]+>[^<>]+<\\/[^<>]+>',xmlList)[[1]])
    pattern_length <- as.vector(attr(gregexpr('<[^/<>]+>[^<>]+<\\/[^<>]+>',xmlList)[[1]],'match.length'))
    pattern_end <- pattern_start+pattern_length-1
    
    xml_data = rep(NA,length(pattern_start))
    for(i in 1:length(pattern_start)){
        xml_data[i] <- substr(xmlList,pattern_start[i],pattern_end[i])
    }
    
    return(xml_data)
}

#diag_Processer(�� �Ʒ����� �±��� ������ �̰� ù �±׺��� ���� ù�±׷� ������.)
#ex) EMR ���   ���    ##��
#    EMR ���ܸ� ���ܸ�  ����

#�Ľ��ϸ� �� ����� ?? ���ܸ��� ??�� �����ϱ� ���� �Լ�.
DIAG_PROCESSING <- function(diag_list){
    #ù��° > �� �������� �±׸� ����
    tag_vector  <- as.vector(regexpr('>',diag_list))
    text_vector <- as.vector(regexpr('</',diag_list))
    
    #tag�� text�� ������ �����
    tag_data_vector <- substr(diag_list,1,tag_vector)
    text_data_vector <- substr(diag_list,tag_vector+1,text_vector-1)
    
    #ù��° �±׺��� ���� ù �±ױ��� ���� ���� �Ҵ�
    first_tag_vector <- as.vector(regexpr(tag_data_vector[1],tag_data_vector))# ù��° ������ ������ 1�� ��ȯ
    
    #1�� ��ġ�� ã�� ��ġ ���� �־���
    data =c()
    for (i in 1:length(first_tag_vector)){
        if (first_tag_vector[i] == 1){
            data[i] <- i
        }
    }
    #NA ����
    data <- data[!is.na(data)]
    
    #������ ù��° �±��� ������ ��
    data[length(data)+1] <- length(first_tag_vector)+1
    
    #tag�� ���̸� ����
    df <- data.frame(stringsAsFactors = FALSE)
    for (i in unique(tag_data_vector)){
        df[i] <- character(0)
    }
    
    #df�� �� �ֱ�
    cnt <- 1
    for (i in 1:(length(data)-1)){
        val <- (data[i+1])-(data[i])
        for (k in 1:val){
            df[i,tag_data_vector[cnt]] <- text_data_vector[cnt]
            cnt <- cnt+1
        }
    }
    
    return(df)
}


########MAIN CODE##############################################################


# load packages
if(!require(parallel)) {
    install.packages("parallel")
}
library(parallel)

# �ھ� ���� ȹ��
numCores <- parallel::detectCores() - 1
# Ŭ������ �ʱ�ȭ
myCluster <- parallel::makeCluster(numCores)

Sys.time()

connectionDetails<-DatabaseConnector::createConnectionDetails(dbms="sql server",
                                                              server="128.1.99.58",
                                                              schema="Dolphin_CDM.dbo",
                                                              user="atlas",
                                                              password="qwer1234!@")
connection <- DatabaseConnector::connect(connectionDetails)
connectionDetails <-connectionDetails
connection <- connection



#
diag_note <- DatabaseConnector::dbGetQuery(conn = connection,statement = "SELECT TOP 100 * FROM DBO.NOTE JOIN COHORT ON NOTE.person_id = COHORT.subject_id AND NOTE.NOTE_DATE = COHORT.COHORT_START_DATE WHERE cohort_definition_id = 747 AND NOTE_TITLE = \'������\'") ;

#���� ���� �����ϴ� df���� merge �� ����###############################################
cohort_outCount_df <- merge(outcomeCount_df,diag_note,by = c("PERSON_ID","NOTE_DATE"))
#######################################################################################

#�ʿ��� ���� �������� df ����
cohort_outCount_df <- data.frame(c(cohort_outCount_df['NOTE_ID'],cohort_outCount_df['NOTE_TEXT'],cohort_outCount_df['outcomeCount']),stringsAsFactors = FALSE)

#XML �ļ��� ����(����ó��)
diagnosis_list <- parallel::parLapply(cl = myCluster, X = cohort_outCount_df$NOTE_TEXT, fun = XML_PARSING)

#��� ������ df ����
final_xml_df <- data.frame(stringsAsFactors = FALSE) 

#���ܼ� �ϳ��� DataFrame�� list�� ����(����ó��)
result_xml_list <- parallel::parLapply(cl = myCluster, X = diagnosis_list, fun = DIAG_PROCESSING)

#�Ѱ��� ���ܼ��� NOTE_ID ����
for (i in 1:length(result_xml_list)){
    result_xml_list[[i]][,'NOTE_ID'] <- cohort_outCount_df[['NOTE_ID']][i]
    result_xml_list[[i]][,'outcomeCount'] <- cohort_outCount_df[['outcomeCount']][i]
}
    
#���� ū ���� ã��
max_col <- 0
for(i in 1:length(result_xml_list)){
    col_value <- length(result_xml_list[[i]])
    if(max_col < col_value){
        max_col <- col_value
    }
}


result_tmp_df <- data.frame(stringsAsFactors = FALSE)
result_tmp2_df <- data.frame(stringsAsFactors = FALSE)
result_xml_df <- data.frame(stringsAsFactors = FALSE)
div = 1000
flag <- 0
if(div >= length(result_xml_list)){
    for(i in 1:length(result_xml_list)){
        if(length(result_xml_list[[i]]) == max_col){
            result_tmp_df <- rbind(result_tmp_df,result_xml_list[[i]])     
        }
    }
    flag <- 1
    result_xml_df <- result_tmp_df
}
if(flag == 0 ){
    for(i in 1:length(result_xml_list)){
        
        if(length(result_xml_list[[i]]) == max_col){
            
            result_tmp_df <- rbind(result_tmp_df,result_xml_list[[i]])
            
            if(i%%div ==0 & i>=div){
                result_xml_df <- rbind(result_xml_df,result_tmp_df)
                result_tmp_df <- data.frame(stringsAsFactors = FALSE)
            }
        }
        else{
            # �ٸ� �͵�� �ٸ� �÷� ������ ������ �ִ� �ֵ��� ��� ó�������� �����غ�����.
        }
    }
    
    result_xml_df <- rbind(result_xml_df,result_tmp_df)
}
