# Find Smiliar TPO 

In TOEFL learning, we can find many TPO materials are similar. Even some reading materials and listening materials come from the same long articles. This is a small project to help you find the similar TPO materials to improve you learning efficiency.

Using gensim package calculate the cosine similiarity of the TPO materials.

## Environment
The code is developed under the following configurations.
- Software: Python>=3.5
- Dependencies: gensim, jieba, collections, heapq, argparse

## Quick start
To find the similar material with one TPO, you can simply do the following:

```
python3 FindSimilarTPO.py --filename 1-1.txt --num 10 --tpo 54
python3 FindSimilarTPO.py -f 1-1.txt -n 10 -t 54 
#find Reading material TPO1-1
python3 FindSimilarTPO.py -f L54-4.txt --n 5
#find Listening material TPO54-lecture 4
python3 FindSimilarTPO.py -f C54-2.txt 
#find Listening material TPO54-conversation 2
```

* "num" is records number you want to see. 
* "tpo" is tpo numbers in your data.
* Each TPO has 3 reading materials, 4 Letures and 2 conversations.
* This project perform well on reading material and leture. Due to conversation has too much daliy dialogues, the results probably cannot find the similar topic.

Hope you can get a great TOEFL score !




