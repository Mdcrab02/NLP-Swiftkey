setwd("C:/R_Workspace/Capstone_Project")
library(R.utils)
library(stringr)
library(openNLP)
library(tm) 
library(qdap)
library(RWeka)
library(ggplot2)

news <- "./en_US.news.txt"
blogs <- "./en_US.blogs.txt" 
twitter <- "./en_US.twitter.txt"

news_data <- scan(news, character(0), sep = "\n", skipNul = TRUE)
news_lines <- as.numeric(countLines(news))

blogs_data <- scan(blogs, character(0), sep = "\n", skipNul = TRUE)
blogs_lines <- as.numeric(countLines(blogs))

twitter_data <- scan(twitter, character(0), sep = "\n", skipNul = TRUE)
twitter_lines <- as.numeric(countLines(twitter))

print(paste("The line count for the News data is: ", news_lines))
print(paste("The line count for the Blogs data is: ", blogs_lines))
print(paste("The line count for the Twitter data is: ", twitter_lines))

news_con <- file("./en_US.news.txt", "r") 
news_sample <-readLines(news_con,
                        (news_lines/1500),encoding= "UTF-16LE")
writeLines(news_sample, con="datanewsa.txt", "\n")
close(news_con)

blogs_con <- file("./en_US.blogs.txt", "r") 
blog_sample <-readLines(blogs_con,
                        (blogs_lines/1500),encoding= "UTF-16LE")
writeLines(blog_sample, con="databloga.txt", "\n")
close(blogs_con)

twitter_con <- file("./en_US.twitter.txt", "r")
twitter_sample <- readLines(twitter_con,
                            (twitter_lines/1500),encoding= "UTF-16LE")
writeLines(twitter_sample, con="datatwa.txt", "\n")
close(twitter_con)

data_all <- paste(blog_sample,news_sample,twitter_sample)  
data_all <- sent_detect(data_all, language = "en", model = NULL)

corpus <- VCorpus(VectorSource(data_all))

corpus <- tm_map(corpus, content_transformer(tolower)) 
corpus <- tm_map(corpus, content_transformer(removePunctuation))
corpus <- tm_map(corpus, content_transformer(removeNumbers))
corpus <- tm_map(corpus, content_transformer(stripWhitespace))

url_fun <- function(x){
  gsub("http[[:alnum:]]*", "", x)  
}
corpus <- tm_map(corpus, content_transformer(url_fun))

profanity <- readLines("./bad-words.txt")

corpus <- tm_map(corpus, removeWords, profanity) 

onegram <- function(x) {
  NGramTokenizer(x, Weka_control(min = 1, max = 1))
}
bigram <- function(x) {
  NGramTokenizer(x, Weka_control(min = 2, max = 2))
}
trigram <- function(x) {
  NGramTokenizer(x, Weka_control(min = 3, max = 3))
}

doc_onegram <- DocumentTermMatrix(corpus
                                  ,control=list(tokenize=onegram))
doc_bigram  <- DocumentTermMatrix(corpus
                                  ,control=list(tokenize=bigram))
doc_trigram <- DocumentTermMatrix(corpus
                                  ,control=list(tokenize=trigram))

onegram_sort <- sort(colSums(as.matrix(doc_onegram))
                     ,decreasing=TRUE)
bigram_sort <- sort(colSums(as.matrix(doc_bigram))
                    ,decreasing=TRUE)
trigram_sort <- sort(colSums(as.matrix(doc_trigram))
                     ,decreasing=TRUE)

barplot(onegram_sort[1:25],
        ylab='Frequency',
        xlab='Word',
        main='Top 25 Most Frequently Used Words',
        names.arg=names(onegram_sort)[1:25],        
        col="blue", las=2, cex.names=.7)

barplot(bigram_sort[1:25],
        ylab='Frequency',
        xlab='Phrase',
        main='Top 25 Most Frequently Used 2-word Phrases',
        names.arg=names(bigram_sort)[1:25],        
        col="green", las=2, cex.names=.7)

barplot(trigram_sort[1:25],
        ylab='Frequency',
        xlab='Phrase',
        main='Top 25 Most Frequently Used 3-word Phrases',
        names.arg=names(trigram_sort)[1:25],        
        col="red", las=2, cex.names=.7)