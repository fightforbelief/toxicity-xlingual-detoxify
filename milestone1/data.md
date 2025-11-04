# Data Description

_Last generated: 2025-11-04T14:43:11_

Root directory: `/Users/alex/Desktop/data`

## Directory Overview

```
├── civil_comments/
  ├── test-00000-of-00001.parquet
  ├── train-00000-of-00002.parquet
  ├── train-00001-of-00002.parquet
  └── validation-00000-of-00001.parquet
├── jigsaw-multilingual-toxic-comment-classification/
  ├── jigsaw-toxic-comment-train-processed-seqlen128.csv
  ├── jigsaw-toxic-comment-train.csv
  ├── jigsaw-unintended-bias-train-processed-seqlen128.csv
  ├── jigsaw-unintended-bias-train.csv
  ├── sample_submission.csv
  ├── test-processed-seqlen128.csv
  ├── test.csv
  ├── test_labels.csv
  ├── validation-processed-seqlen128.csv
  └── validation.csv
└── jigsaw-toxic-comment-classification-challenge/
  ├── sample_submission.csv.zip
  ├── test.csv.zip
  ├── test_labels.csv.zip
  └── train.csv.zip
```

## Datasets Included

### Jigsaw Multilingual Toxic Comment Classification (2020)

- **Description:** Multilingual toxicity dataset (English, Spanish, French, Italian, Portuguese, Russian, Turkish; with German in some test splits).
- **Format:** CSV (UTF-8), comma-delimited.
- **Upstream Links:**  
  - Competition page (download, license): https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification  
  - Baseline code/project context: https://github.com/unitaryai/detoxify
- **Local Files:**

- `jigsaw-toxic-comment-train-processed-seqlen128.csv` — 387.9 MB, modified 2021-09-28T17:35:48
- `jigsaw-toxic-comment-train.csv` — 91.1 MB, modified 2021-09-28T17:36:10
- `jigsaw-unintended-bias-train-processed-seqlen128.csv` — 3.3 GB, modified 2021-09-28T17:36:18
- `jigsaw-unintended-bias-train.csv` — 820.5 MB, modified 2021-09-28T17:39:24
- `sample_submission.csv` — 612.3 KB, modified 2021-09-28T17:40:36
- `test-processed-seqlen128.csv` — 117.3 MB, modified 2021-09-28T17:40:36
- `test.csv` — 27.4 MB, modified 2021-09-28T17:40:44
- `test_labels.csv` — 487.7 KB, modified 2021-09-28T17:40:46
- `validation-processed-seqlen128.csv` — 14.3 MB, modified 2021-09-28T17:40:46
- `validation.csv` — 3.0 MB, modified 2021-09-28T17:40:48

**Example rows — Train (multilingual)**  
_File:_ `jigsaw-toxic-comment-train.csv`

```
              id                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       comment_text  toxic  severe_toxic  obscene  threat  insult  identity_hate
0000997932d777bf                                                                                                                                                                                                                                                                                                                                                                          Explanation\nWhy the edits made under my username Hardcore Metallica Fan were reverted? They weren't vandalisms, just closure on some GAs after I voted at New York Dolls FAC. And please don't remove the template from the talk page since I'm retired now.89.205.38.27      0             0        0       0       0              0
000103f0d9cfb60f                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   D'aww! He matches this background colour I'm seemingly stuck with. Thanks.  (talk) 21:51, January 11, 2016 (UTC)      0             0        0       0       0              0
000113f07ec002fd                                                                                                                                                                                                                                                                                                                                                                                                          Hey man, I'm really not trying to edit war. It's just that this guy is constantly removing relevant information and talking to me through edits instead of my talk page. He seems to care more about the formatting than the actual info.      0             0        0       0       0              0
0001b41b1c6bb37e "\nMore\nI can't make any real suggestions on improvement - I wondered if the section statistics should be later on, or a subsection of ""types of accidents""  -I think the references may need tidying so that they are all in the exact same format ie date format etc. I can do that later on, if no-one else does first - if you have any preferences for formatting style on references or want to do it yourself please let me know.\n\nThere appears to be a backlog on articles for review so I guess there may be a delay until a reviewer turns up. It's listed in the relevant form eg Wikipedia:Good_article_nominations#Transport  "      0             0        0       0       0              0
0001d958c54c6e35                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                You, sir, are my hero. Any chance you remember what page that's on?      0             0        0       0       0              0
```

**Inferred columns/dtypes (sample):**
- `id`: object
- `comment_text`: object
- `toxic`: int64
- `severe_toxic`: int64
- `obscene`: int64
- `threat`: int64
- `insult`: int64
- `identity_hate`: int64


**Example rows — Validation**  
_File:_ `validation.csv`

```
 id                                                                                                                                                                                                                                                                                                                                                                                                                               comment_text lang  toxic
  0                                                                                                                                                                               Este usuario ni siquiera llega al rango de    hereje   . Por lo tanto debería ser quemado en la barbacoa para purificar su alma y nuestro aparato digestivo mediante su ingestión.    Skipe linkin 22px   Honor, valor, leltad.      17:48 13 mar 2008 (UTC)   es      0
  1                                                                                                                                                                                                                                                                                                         Il testo di questa voce pare esser scopiazzato direttamente da qui. Immagino possano esserci problemi di copyright, nel fare cio .   it      0
  2                                                                                                                                                                                                                                                                     Vale. Sólo expongo mi pasado. Todo tiempo pasado fue mejor, ni mucho menos, yo no quisiera retroceder 31 años a nivel particular. Las volveria a pasar putas.Fernando    es      1
  3                                  Bu maddenin alt başlığı olarak  uluslararası ilişkiler  ile konuyu sürdürmek ile ilgili tereddütlerim var.Önerim siyaset bilimi ana başlığından sonra siyasal yaşam ve toplum, siyasal güç, siyasal çatışma, siyasal gruplar, çağdaş ideolojiler, din, siyasal değişme, kamuoyu, propaganda ve siyasal katılma temelinde çoğulcu siyasal sistemler.Bu alt başlıkların daha anlamlı olacağı kanaatindeyim.   tr      0
  4 Belçika nın şehirlerinin yanında ilçe ve beldelerini yaparken sanırım Portekizi örnek alacaksın. Ben de uzak gelecekte(2-3 yıl) bu tip şeyler düşünüyorum. Tabii futbol maddelerinin hakkından geldikten sonra..    daha önce mesajlarınızı görmüştüm, hatta anon bölümünü bizzat kullanıyordum   sözünü anlamadım??  tanışmak bugüneymiş gibi bir şey eklemeyi düşündüm ama vazgeçtim. orayı da silmeyi unuttum. boşverin Kıdemli   +       tr      0
```

**Inferred columns/dtypes (sample):**
- `id`: int64
- `comment_text`: object
- `lang`: object
- `toxic`: int64


**Example rows — Test (no labels)**  
_File:_ `test.csv`

```
 id                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    content lang
  0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              Doctor Who adlı viki başlığına 12. doctor olarak bir viki yazarı kendi adını eklemiştir. Şahsen düzelttim. Onaylarsanız sevinirim. Occipital    tr
  1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Вполне возможно, но я пока не вижу необходимости выделять материал в отдельную статью. Если про правосудие в СССР будет написано хотя бы килобайт 20-30 — тогда да, следует разделить. Пока же мы в итоге получим одну куцую статью Правосудие и другую не менее куцую статью Правосудие в СССР. Мне кажется, что этот вопрос вполне разумно решать на основе правил ВП:Размер статей? которые не предписывают разделения, пока размер статьи не достигнет хотя бы 50 тыс. знаков.    ru
  2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              Quindi tu sei uno di quelli   conservativi  , che preferiscono non cancellare. Ok. Avresti lasciato anche   sfaccimma  ? Si? Ok. Contento te... io non approvo per nulla, ma non conto nemmeno nulla... Allora lo sai che faccio? Me ne frego! (Aborro il fascismo, ma quando ce vo , ce vo !) Elborgo (sms)    it
  3                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   Malesef gerçekleştirilmedi ancak şöyle bir şey vardı. Belki yararlanırsınız. İyi çalışmalar.    Kud      yaz     Teşekkür ederim. Abidenin maddesini de genişletmeyi düşünüyorum, ileride işime yarayacak bu. cobija  Kullandın mı bilmiyorum ama şunu ve şunu da ben iliştireyim. Belki kaynakçaları lazım olur )RapsarEfendim?  Yok mu artıran? ) . Kullandınız mı bilmiyorum ama kullanmadıysanız alttaki model, 3d, senaryo ve yerleştirme başlıklarını da incelemenizi tavsiye ederim.    Kud      yaz     Aynen ya, çok güzel bir kaynak ama çalışma sahiplerine attığım e-postaya bir cevap gelmedi. Oradaki çalışmaları kullanabilseydim güzel olacaktı. cobija    tr
  4 :Resim:Seldabagcan.jpg resminde kaynak sorunu    :Resim:Seldabagcan.jpg    resmini yüklediğiniz için teşekkürler. Ancak dosyanın tanım sayfasında içeriğin kimin tarafından yapıldığı hakkında ayrıntılı bilgi bulunmamaktadır, yani telif durumu açık değildir. Eğer dosyayı kendiniz yapmadıysanız, içeriğin sahibini belirtmelisiniz. Bir internet sitesinden elde ettiyseniz nereden aldığınızı net şekilde gösteren bir bağlantı veriniz. Diğer yüklediğiniz resimleri kontrol etmek istiyorsanız bu bağlantıyı tıklayın.    Kaynaksız ve lisanssız resimler hızlı silme kriterlerinde belirtildiği üzere işaretlendikten bir hafta sonra silinirler.    Telif hakları saklı olup adil kullanım politikasına uymayan resimler    48 saat sonra silinirler   . Sorularınız için Vikipedi:Medya telif soruları sayfasını kullanabilirsiniz. Teşekkürler.    Yabancı     msj    :Resim:Seldabagcan.jpg için adil kullanım gerekçesi          :Resim:Seldabagcan.jpg    resmini yüklediğiniz için teşekkürler. Yüklediğiniz resim adil kullanım politikasına uymak zorundadır ancak bu politikaya nasıl uyduğunu gösteren bir açıklama veya gerekçe bulunmamaktadır. Resim tanım sayfasına, kullanıldığı her madde için ayrı ayrı olacak şekilde bir    adil kullanım gerekçesi    yazmalısınız. Yüklediğiniz diğer resimleri kontrol etmek için bu bağlantıyı tıklayınız.    Gerekçesi eksik olan adil kullanım resimleri hızlı silme kriterleri gereğince bir hafta sonra silinirler.    Sorularınız için Vikipedi:Medya telif soruları sayfasını kullanabilirsiniz. Teşekkürler.    Yabancı     msj      tr
```

**Inferred columns/dtypes (sample):**
- `id`: int64
- `content`: object
- `lang`: object


**Example rows — English (Civil Comments subset with identity labels)**  
_File:_ `jigsaw-unintended-bias-train.csv`

```
   id                                                                                                       comment_text    toxic  severe_toxicity  obscene  identity_attack  insult  threat
59848              This is so cool. It's like, 'would you want your mother to read this??' Really great idea, well done! 0.000000         0.000000      0.0         0.000000 0.00000     0.0
59849 Thank you!! This would make my life a lot less anxiety-inducing. Keep it up, and don't let anyone get in your way! 0.000000         0.000000      0.0         0.000000 0.00000     0.0
59852                             This is such an urgent design problem; kudos to you for taking it on. Very impressive! 0.000000         0.000000      0.0         0.000000 0.00000     0.0
59855                               Is this something I'll be able to install on my site? When will you be releasing it? 0.000000         0.000000      0.0         0.000000 0.00000     0.0
59856                                                                               haha you guys are a bunch of losers. 0.893617         0.021277      0.0         0.021277 0.87234     0.0
```

**Inferred columns/dtypes (sample):**
- `id`: int64
- `comment_text`: object
- `toxic`: float64
- `severe_toxicity`: float64
- `obscene`: float64
- `identity_attack`: float64
- `insult`: float64
- `threat`: float64
- `asian`: float64
- `atheist`: float64
- `bisexual`: float64
- `black`: float64
- `buddhist`: float64
- `christian`: float64
- `female`: float64
- `heterosexual`: float64
- `hindu`: float64
- `homosexual_gay_or_lesbian`: float64
- `intellectual_or_learning_disability`: float64
- `jewish`: float64
- `latino`: float64
- `male`: float64
- `muslim`: float64
- `other_disability`: float64
- `other_gender`: float64
- `other_race_or_ethnicity`: float64
- `other_religion`: float64
- `other_sexual_orientation`: float64
- `physical_disability`: float64
- `psychiatric_or_mental_illness`: float64
- `transgender`: float64
- `white`: float64
- `created_date`: object
- `publication_id`: int64
- `parent_id`: float64
- `article_id`: int64
- `rating`: object
- `funny`: int64
- `wow`: int64
- `sad`: int64
- `likes`: int64
- `disagree`: int64
- `sexual_explicit`: float64
- `identity_annotator_count`: int64
- `toxicity_annotator_count`: int64


**Example rows — Processed Train (seq len 128)**  
_File:_ `jigsaw-toxic-comment-train-processed-seqlen128.csv`

```
              id                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       comment_text  toxic  severe_toxic  obscene  threat  insult  identity_hate
0000997932d777bf                                                                                                                                                                                                                                                                                                                                                                          Explanation\nWhy the edits made under my username Hardcore Metallica Fan were reverted? They weren't vandalisms, just closure on some GAs after I voted at New York Dolls FAC. And please don't remove the template from the talk page since I'm retired now.89.205.38.27      0             0        0       0       0              0
000103f0d9cfb60f                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   D'aww! He matches this background colour I'm seemingly stuck with. Thanks.  (talk) 21:51, January 11, 2016 (UTC)      0             0        0       0       0              0
000113f07ec002fd                                                                                                                                                                                                                                                                                                                                                                                                          Hey man, I'm really not trying to edit war. It's just that this guy is constantly removing relevant information and talking to me through edits instead of my talk page. He seems to care more about the formatting than the actual info.      0             0        0       0       0              0
0001b41b1c6bb37e "\nMore\nI can't make any real suggestions on improvement - I wondered if the section statistics should be later on, or a subsection of ""types of accidents""  -I think the references may need tidying so that they are all in the exact same format ie date format etc. I can do that later on, if no-one else does first - if you have any preferences for formatting style on references or want to do it yourself please let me know.\n\nThere appears to be a backlog on articles for review so I guess there may be a delay until a reviewer turns up. It's listed in the relevant form eg Wikipedia:Good_article_nominations#Transport  "      0             0        0       0       0              0
0001d958c54c6e35                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                You, sir, are my hero. Any chance you remember what page that's on?      0             0        0       0       0              0
```

**Inferred columns/dtypes (sample):**
- `id`: object
- `comment_text`: object
- `toxic`: int64
- `severe_toxic`: int64
- `obscene`: int64
- `threat`: int64
- `insult`: int64
- `identity_hate`: int64
- `input_word_ids`: object
- `input_mask`: object
- `all_segment_id`: object


**Notes on processed files (`*-processed-seqlen128.csv`):**  
These files were produced by our preprocessing pipeline (lowercasing, Unicode normalization, punctuation cleanup, and tokenization with XLM-R tokenizer), then truncated/padded to a maximum sequence length of 128. They keep the same core fields (e.g., `comment_text`/`toxic`/`lang` where applicable), plus any auxiliary fields required for faster training.


### Civil Comments (Jigsaw 2019 Unintended Bias)

- **Description:** Large English comment corpus with toxicity and identity-related labels; supports bias-aware metrics.
- **Format:** Parquet (columnar).
- **Upstream Links:**  
  - Official GCS archive: https://storage.googleapis.com/jigsaw-unintended-bias-in-toxicity-classification/civil_comments.zip  
  - TFDS catalog reference: https://www.tensorflow.org/datasets/catalog/civil_comments
- **Local Files:**

- `test-00000-of-00001.parquet` — 19.8 MB, modified 2025-10-31T16:34:57
- `train-00000-of-00002.parquet` — 184.6 MB, modified 2025-10-31T16:35:33
- `train-00001-of-00002.parquet` — 178.1 MB, modified 2025-10-31T16:35:25
- `validation-00000-of-00001.parquet` — 20.0 MB, modified 2025-10-31T16:35:17

**Example rows — Civil Comments shard**  
_File:_ `train-00000-of-00002.parquet`

> Parquet engine not installed (pyarrow/fastparquet).


**Example rows — Civil Comments shard**  
_File:_ `train-00001-of-00002.parquet`

> Parquet engine not installed (pyarrow/fastparquet).


**Example rows — Civil Comments shard**  
_File:_ `validation-00000-of-00001.parquet`

> Parquet engine not installed (pyarrow/fastparquet).


**Example rows — Civil Comments shard**  
_File:_ `test-00000-of-00001.parquet`

> Parquet engine not installed (pyarrow/fastparquet).


**Parquet rationale:** Columnar storage reduces disk footprint and accelerates selective column reads (e.g., `['comment_text','toxicity','identity_attack']`) during training and analysis.


### Jigsaw Toxic Comment Classification Challenge (2018)

- **Description:** English multi-label toxicity task (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`).
- **Format:** CSV inside ZIP archives (UTF-8).
- **Upstream Link:** https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge
- **Local Files:**

- `sample_submission.csv.zip` — 1.4 MB, modified 2019-12-11T04:00:40
- `test.csv.zip` — 23.4 MB, modified 2019-12-11T04:00:42
- `test_labels.csv.zip` — 1.5 MB, modified 2019-12-11T04:00:52
- `train.csv.zip` — 26.3 MB, modified 2019-12-11T04:00:52

> Note: These ZIPs contain `train.csv`, `test.csv`, `test_labels.csv`, and a `sample_submission.csv`.  
> We keep them compressed as a historical English-only baseline and to ensure licensing terms from Kaggle are respected.


## File Formats


- **CSV (UTF-8, comma-delimited):** Used by the Jigsaw 2018/2020 releases. Typical columns include an ID, free-text (`comment_text`), one or more labels (e.g., `toxic`), and sometimes a language code (`lang`).  
- **Parquet:** Columnar storage used for Civil Comments shards; preserves the same labels plus identity attributes for bias-aware evaluation.
- **Processed CSV (`*-processed-seqlen128.csv`):** Our pipeline outputs (deterministic text cleaning + tokenization + truncation to 128 tokens) to speed up training.


## Links to Full Datasets


- Jigsaw Multilingual Toxic Comment Classification (2020): https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification  
- Jigsaw Toxic Comment Classification Challenge (2018): https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge  
- Civil Comments (2019) official archive (GCS): https://storage.googleapis.com/jigsaw-unintended-bias-in-toxicity-classification/civil_comments.zip  
- TFDS catalog (Civil Comments reference): https://www.tensorflow.org/datasets/catalog/civil_comments


## Provenance and Collection Process


Data were collected from official competition pages (Kaggle, 2018/2020) and the publicly hosted Civil Comments archive (Google Cloud Storage) referenced by TFDS.  
Multilingual processed files (`*-processed-seqlen128.csv`) were created locally by our preprocessing script (lowercasing, Unicode normalization, punctuation cleanup, XLM-R tokenization) to ensure fixed-length inputs for model training.
