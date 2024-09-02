Language-Change-CH/  
├── alignment/ *(training the alignment-based temporal embeddings)*  
│   ├── 1-year/ *(1-year time slices)*  
│   │   └── **train.ipynb**  
│   ├── 2-slices/ *(1954-1978 vs 1979-2003)*  
│   │   └── **train.ipynb**  
│   ├── 5-year/ *(5-year time slices)*  
│   │   └── **train.ipynb**  
│   ├── example/ *(mini example for English)*  
│   │   ├── training/ *(example training data)*  
│   │   │   ├── arxiv_14.txt  
│   │   │   └── arxiv_9.txt  
│   │   └── **example.ipynb**  
│   └── hist/ *(source code for alignment-based model)*  
│       └── hist.py  
├── bert/ *(still under construction, ignore for now)*  
├── compass/ *(training the compass-based temporal embeddings)*  
│   ├── 1-year/ *(1-year time slices)*  
│   │   └── **train.ipynb**  
│   ├── 2-slices/ *(1954-1978 vs 1979-2003)*  
│   │   └── **train.ipynb**  
│   ├── 5-year/ *(5-year time slices)*  
│   │   └── **train.ipynb**  
│   ├── example/ *(mini example for English)*  
│   │   ├── training/ *(example training data)*  
│   │   │   ├── arxiv_14.txt  
│   │   │   ├── arxiv_9.txt  
│   │   │   └── compass.txt *(the two concatenated)*  
│   │   └── **example.ipynb**  
│   └── twec/ (source code for compass-based model)  
│       └── twec.py  
├── ~~corpus/~~  
│   ├── ~~1-year/~~ *(tokenised 1946.txt ~ 2023.txt)*  
│   ├── ~~2-slices/~~ *(tokenised 2-slices)*  
│   │   ├── ~~1954-1978.txt~~  
│   │   ├── ~~1979-2003.txt~~  
│   │   └── ~~compass.txt~~ *(compass for 2-slices)*  
│   ├── ~~5-year/~~ *(tokenised 1945-1949.txt ~ 2020-2024.txt)*  
│   ├── ~~freq-5/~~ *(frequency count for 5-year)*  
│   ├── ~~frequency~~ *(frequency count for 1-year)*  
│   ├── ~~jan-1971/~~ *(complementary news data from laoziliao)*  
│   ├── ~~raw/~~ *(news data 1946.jsonl ~ 2023.jsonl)*  
│   ├── ~~sent/~~ *(sentences 1946.txt ~ 2023.txt)*  
│   ├── ~~compass.txt~~ *(compass for 1-year and 5-year)*  
│   ├── dict.mdx *(A Dictionary of Current Chinese)*  
│   ├── stopwords.txt *(list of Chinese stopwords)*  
│   ├── ~~vocab_all.txt~~ *(all words in the corpus)*  
│   ├── ~~vocab_common.txt~~ *(alignmen-based common words)*  
│   ├── ~~vocab_filter.txt~~ *(words with count > 100 on 5-year)*  
│   ├── ~~vocab_nonch.txt~~ *(none-Chinese words)*  
│   └── ~~word_count.txt~~ *(total word count each year)*  
├── evaluation/ *(evalutaion tasks)*  
│   ├── analogy/ *(temporal word analogy)*  
│   │   ├── facts/ *(list national leaders' names)*  
│   │   │   └── politicians.csv  
│   │   ├── graphic/ *(plot scores vs time depth)*  
│   │   ├── compare.png *(plot all in one)*  
│   │   ├── **evaluation.ipynb**  
│   │   ├── kr_neighbour.csv *(neighbourhood of kr presidents)*  
│   │   ├── kr_path.png *(visualised path of kr presidents)*  
│   │   ├── politician_freq.csv *(freqency of leaders' names)*  
│   │   ├── south_korea.png *(frequncy of 南朝鲜 vs 韩国)*  
│   │   └── temp_analogy.csv *(temporal analogy scores)*  
│   ├── change/ *(diachronic word similarity)*  
│   │   ├── chi_wug/ *(ChiWUG dataset)*  
│   │   │   └── stats_groupings.csv  
│   │   ├── change_eval.csv *(ChiWUG CHANGE vs cosine dist)*  
│   │   └── **evaluation.ipynb**  
│   └── static/ *(synchronic word sim/word analogy)*  
│       ├── graphic/ *(plot scorse vs year)*  
│       ├── word_analogy *(static analogy testset)*  
│       │   ├── capital.txt  
│       │   ├── city.txt  
│       │   └── family.txt  
│       ├── word_sim *(static word sim testset)*  
│       │   ├── 240.txt  
│       │   └── 297.txt  
│       ├── correlation.csv *(sim scores vs year/word count)*  
│       ├── **evaluation.ipynb**  
│       └── static_eval.csv *(word sim/word analogy scores)*  
├── explore/  
│   ├── analogy/ *(more temporal analogies)*  
│   │   ├── analogy_words/ *(analogy query results)*  
│   │   ├── **analogy.ipynb**  
│   │   ├── 俄乌_2022.png *(plot for analogy 俄乌 2022)*  
│   │   ├── 俄乌_2022.txt *(words vs years of occurrence)*  
│   │   ├── 俄乌_merge.txt *(disambiguation)*  
│   │   ├── 新冠_2022.png *(plot for analogy 新冠 2022)*  
│   │   ├── 新冠_2022.txt *(words vs years of occurrence)*  
│   │   └── 新冠_merge.txt *(disambiguation)*  
│   ├── mining/ *(semantic change mining)*  
│   │   ├── dict_entry.txt *(extracted diction entries)*  
│   │   ├── dist_distribution.png *(distribution of cos dist)*  
│   │   ├── **distance.ipynb** *(calculate dist)*  
│   │   ├── **draw.ipynb** *(draw changing neighbourhood)*  
│   │   ├── filtered_words.txt *(list of filtered words)*  
│   │   ├── judgement.csv *(human annotation)*  
│   │   ├── **neighbours.ipynb** *(list changing neighbourhood)*  
│   │   ├── precision.png *(precision vs dist window)*  
│   │   ├── **statistics.ipynb** *(statistics about changed words)*  
│   │   ├── time_distribution.png *(time of semantic changes)*  
│   │   └── word_distances.csv *(words' cosine distance)*  
│   └── visualisation/ *(visualisation of words' change path)*  
│       ├── change_path/ *(visualised paths)*  
│       └── **visualisation.ipynb**  
├── preprocess/ *(corpus preprocessing)*  
│   ├── raw_stats/ *(stats about raw news data)*  
│   │   ├── raw_stats.csv  
│   │   ├── **raw_stats.ipynb**  
│   │   └── raw_stats.png  
│   ├── sentence/ *(stats about segmented sentences)*  
│   │   ├── **sent_split.ipynb**  
│   │   └── sent_stats.csv  
│   ├── tokenizer/ *(word tokeniser)*  
│   │   ├── tknz_test.ipynb *(testing different tokenisers)*  
│   │   ├── **tokenize.ipynb**  
│   │   └── word_count.png  
│   ├── **mrg_slice.ipynb** *(merge 1-year to 5-year and 2-slices)*  
│   └── **statistics.ipynb** *(stats about tokenised words)*  
├── .gitignore  
└── README.md  