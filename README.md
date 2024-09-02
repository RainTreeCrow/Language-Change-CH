### Analysing Lexical Semantic Change in Chinese Language Using Word Embeddings

#### Project Overview

Our work employed Word2Vec-based temporal word embeddings, designed static and temporal tests to train and evaluate two alignment methods on *People’s Daily* news corpus (1946-2023). We selected a model with balanced performance across tasks for heuristic analysis and exploration, filtered a list of words with potential semantic changes for manual annotation and classification, and used temporal analogy queries to capture real-world events, demonstrating a potential application of temporal word embeddings in sociology and current affairs.

#### Code Structure

<pre><code>Language-Change-CH/
├── alignment/ <i>(training the alignment-based temporal embeddings)</i>
│   ├── 1-year/ <i>(1-year time slices)</i>
│   │   └── <b>train.ipynb</b>
│   ├── 2-slices/ <i>(1954-1978 vs 1979-2003)</i>
│   │   └── <b>train.ipynb</b>
│   ├── 5-year/ <i>(5-year time slices)</i>
│   │   └── <b>train.ipynb</b>
│   ├── example/ <i>(mini example for English)</i>
│   │   ├── training/ <i>(example training data)</i>
│   │   │   ├── arxiv_14.txt
│   │   │   └── arxiv_9.txt
│   │   └── <b>example.ipynb</b>
│   └── hist/ <i>(source code for alignment-based model)</i>
│       └── hist.py
│
├── bert/ <i>(still under construction, ignore for now)</i>
│
├── compass/ <i>(training the compass-based temporal embeddings)</i>
│   ├── 1-year/ <i>(1-year time slices)</i>
│   │   └── <b>train.ipynb</b>
│   ├── 2-slices/ <i>(1954-1978 vs 1979-2003)</i>
│   │   └── <b>train.ipynb</b>
│   ├── 5-year/ <i>(5-year time slices)</i>
│   │   └── <b>train.ipynb</b>
│   ├── example/ <i>(mini example for English)</i>
│   │   ├── training/ <i>(example training data)</i>
│   │   │   ├── arxiv_14.txt
│   │   │   ├── arxiv_9.txt
│   │   │   └── compass.txt <i>(the two concatenated)</i>
│   │   └── <b>example.ipynb</b>
│   └── twec/ <i>(source code for compass-based model)</i>
│       └── twec.py
│
├── <del> corpus/</del>
│   ├── <del> 1-year/</del> <i>(tokenised 1946.txt ~ 2023.txt)</i>
│   ├── <del> 2-slices/</del> <i>(tokenised 2-slices)</i>
│   │   ├── <del> 1954-1978.txt</del>
│   │   ├── <del> 1979-2003.txt</del>
│   │   └── <del> compass.txt</del> <i>(compass for 2-slices)</i>
│   ├── <del> 5-year/</del> <i>(tokenised 1945-1949.txt ~ 2020-2024.txt)</i>
│   ├── <del> freq-5/</del> <i>(frequency count for 5-year)</i>
│   ├── <del> frequency</del> <i>(frequency count for 1-year)</i>
│   ├── <del> jan-1971/</del> <i>(complementary news data from laoziliao)</i>
│   ├── <del> raw/</del> <i>(news data 1946.jsonl ~ 2023.jsonl)</i>
│   ├── <del> sent/</del> <i>(sentences 1946.txt ~ 2023.txt)</i>
│   ├── <del> compass.txt</del> <i>(compass for 1-year and 5-year)</i>
│   ├── dict.mdx <i>(A Dictionary of Current Chinese)</i>
│   ├── stopwords.txt <i>(list of Chinese stopwords)</i>
│   ├── <del> vocab_all.txt</del> <i>(all words in the corpus)</i>
│   ├── <del> vocab_common.txt</del> <i>(alignmen-based common words)</i>
│   ├── <del> vocab_filter.txt</del> <i>(words with count > 100 on 5-year)</i>
│   ├── <del> vocab_nonch.txt</del> <i>(none-Chinese words)</i>
│   └── <del> word_count.txt</del> <i>(total word count each year)</i>
│
├── evaluation/ <i>(evalutaion tasks)</i>
│   ├── analogy/ <i>(temporal word analogy)</i>
│   │   ├── facts/ <i>(list national leaders' names)</i>
│   │   │   └── politicians.csv
│   │   ├── graphic/ <i>(plot scores vs time depth)</i>
│   │   ├── compare.png <i>(plot all in one)</i>
│   │   ├── <b>evaluation.ipynb</b>
│   │   ├── kr_neighbour.csv <i>(neighbourhood of kr presidents)</i>
│   │   ├── kr_path.png <i>(visualised path of kr presidents)</i>
│   │   ├── politician_freq.csv <i>(freqency of leaders' names)</i>
│   │   ├── south_korea.png <i>(frequncy of 南朝鲜 vs 韩国)</i>
│   │   └── temp_analogy.csv <i>(temporal analogy scores)</i>
│   ├── change/ <i>(diachronic word similarity)</i>
│   │   ├── chi_wug/ <i>(ChiWUG dataset)</i>
│   │   │   └── stats_groupings.csv
│   │   ├── change_eval.csv <i>(ChiWUG CHANGE vs cosine dist)</i>
│   │   └── <b>evaluation.ipynb</b>
│   └── static/ <i>(synchronic word sim/word analogy)</i>
│       ├── graphic/ <i>(plot scorse vs year)</i>
│       ├── word_analogy <i>(static analogy testset)</i>
│       │   ├── capital.txt
│       │   ├── city.txt
│       │   └── family.txt
│       ├── word_sim <i>(static word sim testset)</i>
│       │   ├── 240.txt
│       │   └── 297.txt
│       ├── correlation.csv <i>(sim scores vs year/word count)</i>
│       ├── <b>evaluation.ipynb</b>
│       └── static_eval.csv <i>(word sim/word analogy scores)</i>
│
├── explore/
│   ├── analogy/ <i>(more temporal analogies)</i>
│   │   ├── analogy_words/ <i>(analogy query results)</i>
│   │   ├── <b>analogy.ipynb</b>
│   │   ├── 俄乌_2022.png <i>(plot for analogy 俄乌 2022)</i>
│   │   ├── 俄乌_2022.txt <i>(words vs years of occurrence)</i>
│   │   ├── 俄乌_merge.txt <i>(disambiguation)</i>
│   │   ├── 新冠_2022.png <i>(plot for analogy 新冠 2022)</i>
│   │   ├── 新冠_2022.txt <i>(words vs years of occurrence)</i>
│   │   └── 新冠_merge.txt <i>(disambiguation)</i>
│   ├── mining/ <i>(semantic change mining)</i>
│   │   ├── dict_entry.txt <i>(extracted diction entries)</i>
│   │   ├── dist_distribution.png <i>(distribution of cos dist)</i>
│   │   ├── <b>distance.ipynb</b> <i>(calculate dist)</i>
│   │   ├── <b>draw.ipynb</b> <i>(draw changing neighbourhood)</i>
│   │   ├── filtered_words.txt <i>(list of filtered words)</i>
│   │   ├── judgement.csv <i>(human annotation)</i>
│   │   ├── <b>neighbours.ipynb</b> <i>(list changing neighbourhood)</i>
│   │   ├── precision.png <i>(precision vs dist window)</i>
│   │   ├── <b>statistics.ipynb</b> <i>(statistics about changed words)</i>
│   │   ├── time_distribution.png <i>(time of semantic changes)</i>
│   │   └── word_distances.csv <i>(words' cosine distance)</i>
│   └── visualisation/ <i>(visualisation of words' change path)</i>
│       ├── change_path/ <i>(visualised paths)</i>
│       └── <b>visualisation.ipynb</b>
│
├── preprocess/ <i>(corpus preprocessing)</i>
│   ├── raw_stats/ <i>(stats about raw news data)</i>
│   │   ├── raw_stats.csv
│   │   ├── <b>raw_stats.ipynb</b>
│   │   └── raw_stats.png
│   ├── sentence/ <i>(stats about segmented sentences)</i>
│   │   ├── <b>sent_split.ipynb</b>
│   │   └── sent_stats.csv
│   ├── tokenizer/ <i>(word tokeniser)</i>
│   │   ├── tknz_test.ipynb <i>(testing different tokenisers)</i>
│   │   ├── <b>tokenize.ipynb</b>
│   │   └── word_count.png
│   ├── <b>mrg_slice.ipynb</b> <i>(merge 1-year to 5-year and 2-slices)</i>
│   └── <b>statistics.ipynb</b> <i>(stats about tokenised words)</i>
│
├── .gitignore
├── README.md
└── requirements.txt</code></pre>

Large files such as model.npy, .model, and the corpus have not been uploaded.

#### Installation and Setup

Clone the repository
```bash
git clone https://github.com/RainTreeCrow/Language-Change-CH.git
```
Navigate to the project directory
```bash
cd Language-Change-CH
```
Create a virtual environment
```bash
python -m venv virtual
```
Activate the virtual environment
```bash
.\virtual\Scripts\activate
```
Install dependencies
```bash
pip install -r requirements.txt
```

#### Mini Example

Run `example.ipynb` in the `alignment/example` and `compass/example` directories to try out the two models on a mini English corpora (using the trial data presented under the `training` folder).

#### Corpus Download

1. Download the `.jsonl` files from [Hugging Face](https://huggingface.co/datasets/Papersnake/people_daily_news) and place them in the `corpus/raw` directory.

2. Download and extract the `.7z` file from [GitHub](https://github.com/caspiankexin/people-daily-crawler-date/releases/download/1946-2003/1970-1979.7z). Extract the folder `1971年/1971年01月`, place it in the `corpus` directory, and rename the folder to `jan-1971`. This is to complete the missing data.

#### Preprocessing

1. Run `sentence/sent_split.ipynb` in the `preprocess` directory to split sentences in the raw corpus.
2. Run `tokenizer/tokenize.ipynb` to perform word tokenization.
3. Run `mrg_slice.ipynb` to merge 1-year time slices into 5-year and 2-slices.

#### Run the Models

Run `train.ipynb` in the `1-year`, `2-slices` (1954-1978 vs 1979-2003), and `5-year` subfolders under the `alignment` and `compass` directories to train the corresponding models on the respective time slices.

#### Evaluation

1. Run `evaluation.ipynb` in the `static` subfolder under the `evaluation` directory to perform the synchronic word similarity (word relatedness) test and word analogy test.
2. Run `evaluation.ipynb` in the `change` subfolder to conduct diachronic word similarity tests.
3. Run `evaluation.ipynb` in the `analogy` subfolder to carry out temporal word analogy tests.

#### Explore

1. Run `distance.ipynb` and `statistics.ipynb` in the `explore/mining` directory to perform semantic change detection and to conduct statistical analysis on the manually annotated results.
2. Run `analogy.ipynb` in the `explore/analogy` directory to perform additional temporal analogy queries.
3. Run `visualisation.ipynb` in the `explore/visualisation` directory to plot the change trajectories.

#### Customization

Feel free to adjust the corpus or modify the test examples according to your specific needs.

#### Contact

Author: Hanzhi Zhang

Email: hanzhizhang1007@gmail.com
