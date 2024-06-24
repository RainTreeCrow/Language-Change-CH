import os
import logging
import numpy as np
from gensim.models.word2vec import Word2Vec, LineSentence, Vocab


def intersection(m1, m2, words=None):
	"""
	Intersect two gensim word2vec models, m1 and m2.
	Only the shared vocabulary between them is kept.
	If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
	Indices are re-organized from 0..N in order of descending frequency (sum of counts from both m1 and m2).
	These indices correspond to the new syn0 and syn0norm objects in both gensim models:
		-- so that Row 0 of m1.syn0 will be for the same word as Row 0 of m2.syn0
		-- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
	The .vocab dictionary is also updated for each model, preserving the count but updating the index.
	"""

	# Get the vocab for each model
	vocab_m1 = set(m1.wv.vocab.keys())
	vocab_m2 = set(m2.wv.vocab.keys())

	# Find the common vocabulary
	common_vocab = vocab_m1 & vocab_m2
	if words:
		common_vocab &= set(words)

	# If no alignment necessary because vocab is identical...
	if not vocab_m1 - common_vocab and not vocab_m2 - common_vocab:
		return (m1, m2)

	# Otherwise sort by frequency (summed for both)
	common_vocab = list(common_vocab)
	common_vocab.sort(key=lambda w: m1.wv.vocab[w].count + m2.wv.vocab[w].count, reverse=True)

	# Then for each model...
	for m in [m1, m2]:
		# Replace old syn0norm array with new one (with common vocab)
		indices = [m.wv.vocab[w].index for w in common_vocab]
		old_arr = m.wv.syn0norm
		new_arr = np.array([old_arr[index] for index in indices])
		m.wv.syn0norm = m.wv.syn0 = new_arr

		# Replace old vocab dictionary and old index2word with new one (common vocab)
		m.wv.index2word = common_vocab
		old_vocab = m.wv.vocab
		new_vocab = {}
		for new_index, word in enumerate(common_vocab):
			old_vocab_obj = old_vocab[word]
			new_vocab[word] = Vocab(index=new_index, count=old_vocab_obj.count)
		m.wv.vocab = new_vocab

	return (m1, m2)


def procrustes_align(base_embed, other_embed, words=None):
	"""
    Procrustes align two gensim word2vec models (to allow for comparison between same word).
	Code ported from HistWords <https://github.com/williamleif/histwords> by William Hamilton.

	First, intersect the vocabularies.
	Then do the alignment on the other_embed model.
	Replace the other_embed model's syn0 and syn0norm numpy matrices with the aligned version.
	Return other_embed.

	If 'words' is set, intersect the two models' vocabulary with the vocabulary in words.
	"""

	base_embed.init_sims()
	other_embed.init_sims()

	# make sure vocabulary and indices are aligned
	in_base_embed, in_other_embed = intersection(base_embed, other_embed, words=words)

	# get the embedding matrices
	base_vecs = in_base_embed.wv.syn0norm
	other_vecs = in_other_embed.wv.syn0norm

	# just a matrix dot product with numpy
	m = other_vecs.T.dot(base_vecs)
	# SVD method from numpy
	u, _, v = np.linalg.svd(m)
	# another matrix operation
	ortho = u.dot(v)
	# Replace original array with modified one
	# i.e. multiplying the embedding matrix (syn0norm)by "ortho"
	other_embed.wv.syn0norm = other_embed.wv.syn0 = (other_embed.wv.syn0norm).dot(ortho)
	return other_embed


def train_aligned(text, base_embed=None, opath='model',
		size=100, sg=1, it=5, ns=5, window=5, alpha=0.025, min_count=5, workers=3):
	"""
	:param opath: Name of the desired output folder. Default is model.
	:param size: Number of dimensions. Default is 100.
	:param sg: Neural architecture of Word2vec. Default is CBOW (). If 1, Skip-gram is employed.
	:param it: Number of iterations (epochs). Default is 5.
	:param ns: Number of negative sampling examples. Default is 5, min is 1.
	:param window: Size of the context window (left and right). Default is 5 (5 left + 5 right).
	:param alpha: Initial learning rate. Default is 0.025.
	:param min_count: Min frequency for words over the entire corpus. Default is 5.
	:param workers: Number of worker threads. Default is 3.

	If base_embed is not provided, simply train the embedding;
	If provided, train and align.
	"""

	if not os.path.isdir(opath):
		os.makedirs(opath)
	with open(os.path.join(opath, "log.txt"), "w") as f_log:
		f_log.write(str("")) # todo args
		f_log.write('\n')
		logging.basicConfig(filename=os.path.realpath(f_log.name),
			format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	sentences = LineSentence(text)
	embed = Word2Vec(sentences, sg=sg, size=size, alpha=alpha, iter=it,
		negative=ns, window=window, min_count=min_count, workers=workers)

	if base_embed is not None:
		embed = procrustes_align(base_embed, embed)
	model_name = os.path.splitext(os.path.basename(text))[0]

	embed.save(os.path.join(opath, model_name + ".model"))
	return embed