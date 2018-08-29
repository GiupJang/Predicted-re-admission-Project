# Getting the data

install.packages("readr") # read_lines() 사용에 필요.
install.packages("stringr") # str_sub() 사용에 필요.
library(readr)
library(stringr)
reviews <- read_lines("rawfile2.txt") # 문장을 line별로 읽기.
reviews <- iconv(reviews, to="UTF-8") # 인코딩 변경 (필수).
head(reviews, 2) # 추출된 문장 미리보기.

# Preprocessing

install.packages("keras")
library(keras)
library_keras()

install.packages("reticulate")

tokenizer <- text_tokenizer(num_words = 300)
tokenizer %>% fit_text_tokenizer(reviews)

# 이 과정이 없으면 training 단계에서 error 발생

reviews_check <- reviews %>% texts_to_sequences(tokenizer,.) %>% lapply(., function(x) length(x) > 1) %>% unlist(.)
table(reviews_check)
reviews <- reviews[reviews_check]
reviews_check

# Skip-gram model

install.packages("purrr")
library(reticulate)
library(purrr)
skipgrams_generator <- function(text, tokenizer, window_size, negative_samples) {
  gen <- texts_to_sequences_generator(tokenizer, sample(text))
  function() {
    skip <- iter_next(gen) %>%
      skipgrams(
        vocabulary_size = tokenizer$num_words, 
        window_size = window_size, 
        negative_samples = 1
      )
    x <- transpose(skip$couples) %>% map(. %>% unlist %>% as.matrix(ncol = 1))
    y <- skip$labels %>% as.matrix(ncol = 1)
    list(x, y)
  }
}

embedding_size <- 128
skip_window <- 5
num_sampled <- 1

input_target <- layer_input(shape = 1)
input_context <- layer_input(shape = 1)

embedding <- layer_embedding(
  input_dim = tokenizer$num_words + 1, 
  output_dim = embedding_size, 
  input_length = 1, 
  name = "embedding"
)

target_vector <- input_target %>% 
  embedding() %>% 
  layer_flatten()

context_vector <- input_context %>%
  embedding() %>%
  layer_flatten()

dot_product <- layer_dot(list(target_vector, context_vector), axes = 1)
output <- layer_dense(dot_product, units = 1, activation = "sigmoid")

model <- keras_model(list(input_target, input_context), output)
model %>% compile(loss = "binary_crossentropy", optimizer = "adam")

summary(model)

# Model training

model %>%
  fit_generator(
    skipgrams_generator(reviews, tokenizer, skip_window, negative_samples), 
    steps_per_epoch = 100, epochs = 2
    ) # 현재 환경에서 대략 10시간 소요됨. GPU 부재 및 낮은 PC 사양이 원인으로 파악됨.

install.packages("dplyr")
library(dplyr)

embedding_matrix <- get_weights(model)[[1]]

# 여기서 한글 깨짐 발생?
words <- data_frame(
  word = names(tokenizer$word_index), 
  id = as.integer(unlist(tokenizer$word_index))
)

words <- words %>%
  filter(id <= tokenizer$num_words) %>%
  arrange(id)

Encoding(words$word) = "UTF-8" # 필수!!!!!

row.names(embedding_matrix) <- c("UNK", words$word)

# Understanding the embedding

install.packages("text2vec")
library(text2vec)

install.packages("Rtsne")
install.packages("ggplot2")
install.packages("plotly")

library(Rtsne)
library(ggplot2)
library(plotly)

tsne <- Rtsne(embedding_matrix[2:300,], perplexity = 50, pca = FALSE)

tsne_plot <- tsne$Y %>%
  as.data.frame() %>%
  mutate(word = row.names(embedding_matrix)[2:300]) %>%
  ggplot(aes(x = V1, y = V2, label = word)) + 
  geom_text(size = 3)

tsne_plot
