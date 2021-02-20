## 监督学习

- 成对的输入和预期输出提供给算法
- 一般分为两类问题：**分类**与**回归**问题

## 无监督学习

- 只有输入数据是已知的，没有为算法提供输出数据
- Examples: 确定一系列博客文章的主题，将客户分成具有相似偏好的群组，检测网站的异常访问模式



## 监督学习

- 分类：预测类别标签
  - 二分类：正类和反类
- 回归：预测一个连续值或浮点数（根据教育水平、年龄预测一个人的年收入）
- 泛化：对没见过的数据做出精准预测
- 过拟合：构建一个对现有信息量来说过于复杂的模型 -> 预测训练集表现好，不能很好泛化到测试集中
- 欠拟合：选择过于简单的模型被称为欠拟合
- 收集更多数据，适当构建更复杂的模型，对监督学习任务往往更加有用



### 一般步骤

1. 根据数据建立样本数据集（mglearn）
2. 创建训练集与测试集（train_test_split）
3. 构建算法模型实例，并输入训练集数据（fit）
4. predict或score的方法测试测试集数据并验证模型可靠性



### 监督学习算法

#### 一些样本数据集

1. forge: 二分数据集，两个特征
2. wave: 回归数据集，两个特征
3. 威斯康星州乳腺癌数据集： 二分数据集，30个特征
4. 波士顿房价数据集：回归数据集， 13个特征
   1. 特征工程：将每两个特征的乘积作为新的特征（从已有特征导出新的特征）
5. 鸢尾花



#### K邻近

- K邻近分类
  - 邻居少 -> 模型过于复杂 -> 训练集精度高但是泛化程度不行
- K邻近回归
  - **$R^2$**: 决定系数，0到1之间，1 -> 完美预测，0 -> 常数模型
- 优点：模型容易理解
- 缺点：对于很多特征的数据集以及大多数特征取值都为0的数据集（**稀疏数据集**）效果不好



#### 线性模型

- 看做特征的加权求和



1. 线性回归（最小二乘法）

- 寻找w和b, 使得对训练集的预测值与真实的回归目标值y之间的均方误差最小 （https://zh.wikipedia.org/wiki/%E5%9D%87%E6%96%B9%E8%AF%AF%E5%B7%AE）

- 无参数，无法控制模型的复杂度
- 在处理高维（多特征）数据集，存在**过拟合**问题 -> 引出**岭回归**



2. 岭回归

- 希望w尽量小，每一项接近0 -> 每个特征对输出的影响尽可能小

- **L2正则化**： 对模型做显式约束，以避免过拟合

- 调整alpha = 1.0以调整简单性和训练集性能二者对于模型的重要程度； 增大alpha会让W更加趋近0，正则化越强，降低训练集性能，可能会提高泛化性能

  

3. lasso

- **L1正则化**: 某些系数刚好为0 -> 有选择得忽略某些特征，或呈现某些重要的特征
- 训练集和测试集表现都很差 -> **欠拟合** -> 减小alpha，增加max_iter（运行迭代最大次数）的值
- **ElasticNet**: 结合lasso和Ridge，共两个参数



4. 用于分类的线性模型
- y > 0 作为分类标准
   1. Logistic回归（**可延伸到多类**）
   2. 线性支持向量机（**LinearSVC**）

- **C**：正则化强度的权衡参数； C越大，正则化越弱（和alpha正好相反）-> 模型更复杂 -> 避免欠拟合



5. 用于多分类的线性模型 

- **一对多余**：
  - 对每个类别都学习一个二分类模型，将此类别与所有其他类别分开
  - 由此生成了与类别个数一样多的二分类模型。在测试点上运行所有二分类模型进行预测，在对应类别上分数最高的分类器的类别标签返回作为结果



6. 优缺点与参数

- 参数

  - 正则化参数

    - 回归模型：alpha; 

    - LinearSVC，LogisticRegression：C

    - alpha越大或者C值越小：模型越简单，更泛型

  - 正则化

    - L1: 只有几个特征重要 -> 强可解释性， 但是容易**欠拟合**（忽略过多的特征）
    - L2: 默认

- 优点

  - 大型数据处理快，预测速度快，支持稀疏数据
  - - [ ] 研究LogisticRegression和Ridge模型的solver='sag'
  - 理解预测是如何进行的相对容易，但是难以解释**高度相关的特征及其系数**



#### 朴素贝叶斯分类器

- 和线性模型类似，训练速度快，但是泛化能力较弱
- 通过单独查看每个特征来学习参数，并从每个特征中收集简单的类别统计数据
- scikit-learn中的三种分类器
  - GaussionNB：任意连续数据
  - BernoulliNB：二分类数据，计算每个类别中每个特征不为0的元素个数
  - MultinomialNB：计数数据（每个特征代表某个对象的整数计数；**如一个单词在句子里出现的次数**）



#### 决策树

- 从一层层if/else中进行学习



1. 构造决策树

- 学习一系列if/else问题，以最快的速度得到正确答案：**测试**
- 对数据进行反复递归划分，直到每个区域，或者说每个**叶节点**都只包含单一目标值（单一类别或者单一回归值），或者说是**纯**叶节点。



2. 控制决策树的复杂度

- 构造到所有叶节点都是纯的 -> 过于复杂模型，**过拟合**
- 两种策略防止过拟合
  - **预剪枝**：提早停止树的生长；**限制树的最大深度、限制叶节点的最大数目、规定一个节点中数据点的最小数目**
  - **后剪枝**：先构造树，然后删除或折叠信息量少的node；
- scikit-learn中的实现（**只有预剪枝**）
  - DecisionTreeClassifier
  - DecisionTreeRegressor
  - 可以通过设置**max_depth = n**来减少树的最大深度



3. 分析决策树

- tree模块的export_graphviz函数来将树可视化



4. 树的特征重要性

- 0-1之间：0代表没用到，1代表完美预测目标值
- tree.feature_importances_



#### 决策树集成

1. 随机森林

- 原理：通过构造许多略有不同的树，通过去平均值来降低**过拟合**
- 构造随机森林：
  - 首先对数据**自助采样**，从n_sample中有放回地重复随机抽取一个样本，来创建一个相较原数据集缺失一部分数据的新数据集
  - 接着对于这个数据集的每个节点，随机选择特征的一个子集。通过设置max_features，来决定每次选择特征的个数



2. 梯度提升回归树（梯度提升机）

- 原理：采用连续的方式构造树，每棵树都试图纠正前一棵树的错误；使用强预剪枝，深度通常很小
- 参数
  - 预剪枝深度
  - 树的数量
  - *学习率（learning rate)*:控制每棵树纠正前一棵树错误的强度；越高模型越复杂；



#### 核向量支持机SVM

- 原理：学习每个训练数据点对于表示两个类别之间的决策边界的重要性
- *支持向量*：位于类别之间边界上的点的总称
- 参数
  - gamma：决定高丝核的大小；gamma越小，决策边界变化慢，模型平滑简单；gamma越大，决策边界变化快，模型复杂，更关注单个数据点
  - C: 正则化参数；增大c，数据点对模型的影响变大，使得决策边界发生弯曲来将数据点分类
  - 默认值：c = 1, gamma = 1/n_features
- *预处理数据*：将所有特征缩放到0-1之间



#### 神经网络

1. 神经网络模型

- 多层感知机(MLP): 广义线性模型执行多层处理
  
  - 单一层：logistic回归可视化
  
    ```python
    display(mglearn.plots.plot_logistic_regression_graph())
    ```
  
    - MLP中多次重复加权求和：首先计算代表中间过程的*隐单元*，再计算隐单元加权求和得到最终结果
  
  - 在计算完每个*隐单元*加权求和之后，对结果应用一个非线性函数：

      - 校正非线性：relu；用于截断小于0的值
      - 正切双曲线：tanh；输入值较小时接近-1，输入值较大时接近1

- 最后用这个结果加权求和，计算得到输出y

- 小型神经网络举例

  <img src="/Users/mingxulu/Documents/ML-Notes/assets/IMG_1E8FEE715E7D-1.jpeg" alt="IMG_1E8FEE715E7D-1" style="zoom:67%;" />

  - w是输入与隐单元之间的权重，v是隐单元与输出之间的权重



2. 神经网络调参

- hidden_layer_sizes = [default == 100]：隐层数量
- activation = 'default == relu'：非线性函数
- alpha: L2正则化，防止模型过拟合
- max_iter: 最大迭代次数
- solver = 'default == adam': 决定学习的算法
  - 'adam': 对数据缩放敏感，均值为0，方差为1
  - 'lbfgs'
  - 'sgd



#### 分类器的不确定度估计

1. 决策函数：decision_function

- 返回形状：(n_samples, )，每个样本对应一个浮点数；正值代表对该数据点属于正类的置信程度，负值表示对反类的偏好



2. 预测概率：predict_proba

- 输出每个类别的概率：(n_samples, 2), 每行总和为1（对于二分问题来说）



3. 多分类问题的不确定度

- decision_function
  - shape：(n_samples, n_classes)，每一列对应每个类别的确定度分数
- predict_proba
  - shape：(n_samples, n_classes)，每行总和为1



#### 本章小结

<img src="/Users/mingxulu/Documents/ML-Notes/assets/IMG_69F310865D2D-1.jpeg" alt="IMG_69F310865D2D-1" style="zoom:67%;" />








## 无监督学习

### 无监督学习的类型

- **数据集无监督变换**：创建数据新的表示的算法，使数据更加容易被人或其他机器学习算法所理解。
  - 常见应用1：**降维**；将高维数据转为可视化的二位数据
  - 常见应用2：找到“构成”数据的各个组成部分；如对文本文档集合进行主题提取，可用于追踪社交媒体上的话题讨论
- **聚类**：将数据分成不同的组；与监督学习的分类不同的是，聚类算法并无标签，只是单纯讲类似或者距离接近的点分为一类，而不在乎具体的类别是什么



### 无监督学习的挑战

- 评估算法是否学到了有用的东西；由于算法一般用于不包含标签信息的数据 -> 不知道正确的输出 -> 难以评估表现
- 偏向应用于探索性的目的
- 数据**预处理**与**缩放**尤其重要



### 预处理与缩放

```python
mglearn.plots.plot_scaling()
```

#### 不同类型的预处理

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20201221185912137.png" alt="image-20201221185912137" style="zoom:50%;" />

- StandardScaler: 确保每个特征的平均值为0，方差为1
- RobustScaler: 使用中位数和四分位数位于同一范围，忽略很大不同的数据点**（异常值）**
- MinMaxScaler：移动数据使所有特征刚好位于0~1之间
- Normalizer：对每个数据点进行缩放，使得特征向量的欧式距离等于1



#### 聚类

- 将数据集划分成**簇**(clustering)的任务



1. k均值聚类

- 交替进行一下两个步骤
  - 将每个数据点分配给最近的**簇中心**
  - 讲每个**簇中心**设置为所分配的所有数据点的平均值
- 如果簇的分配不再发生变化，那么算法结束。
- **失败案例**
  - 每个簇都是凸型 -> 只能找到相对简单的形状
  - 假设所有簇在某种程度上具有相同的直径 -> 他总是将簇之间的边界刚好画在簇中心的中间文职
  - 假设所有方向对每个簇都同等重要 （分开的三部分被沿着对角线方向拉长）





## 数据表示与特征工程

- 连续特征（continuous feature）
- 离散/分类特征（categorical feature）



### 4.1 分类变量

- 对于logisitic，我们需要将**分类特征**变为**连续特征**才能填入x[p]中

#### 4.1.1 One-Hot编码（虚拟变量）

- 表示分类变量的方法：one-hot编码（one-hot-encoding）或 N取一编码（one-out-of-N encoding），也叫虚拟变量（dummy variable）
  - **将一个分类变量变化为多个“子”特征，当分类变量取某个值，这些“子”特征其中之一取1，其他取0**

- 实施方法：pandas or scikit-learn
  - pandas.get_dummies: 自动变换所有具有对象类型，比如字符串的列或所有分类的列（groupby），**但是会将所有数字看为连续变量**

#### 4.1.2 数字可以编码分类变量

- scikit-learn的OneHotEncoder

- 或者将数值列转化为字符串

  ```python
  df['int feature'] = df['int feature'].astype(str)
  ```



### 4.2 分箱、离散化、线性模型与树

- **分箱**（binning，也叫离散化，discretization）：使线性模型在连续数据上变得更加强大

  - 假设特征的输入范围划分为固定个数的**箱子**（bin），数据点就可以用它所在箱子表示。这里将单个连续输入特征变换为一个分类特征

    ```python
    bins = np.linspace(-3,3,11) # 创建10个箱子，11是连续边界之间的空间
    which_bin = np.digitize(X, bins=bins) # 决定每个数据点所属的箱子
    ```

  - 使用preprocessing模块的onehotencoder将这个离散特征变换为one-hot编码，onehotencoder只适用于**整数**的分类变量

    ```python
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse=False)
    # 使用encoder.fit找到which_bin中的唯一值
    encoder.fit(which_bin)
    # transform创建one-hot编码
    X_binned = encoder.transform(which_bin)
    ```

- **分箱多用于有充分理由使用线性模型，但是其中有些特征与输出的关系是非线性的数据集**



### 4.3 -4.4 TBD



### 4.5 自动化特征选择

- 添加更多特征会使所有模型变得更加复杂 -> 增加过拟合的可能性

#### 4.5.1 单变量统计

- 计算每个特征和目标值之间的关系是否存在统计显著性，并选择具有最高置信度的特征。对于分类问题，也称为方差分析（ANOVA）
- scikit-learn中
  - 分类：``f_classif``
  - 回归：``f_regression``
- 给予测试中确定的p值来选择舍弃特征的方法。所有方法都适用阈值来舍弃所有p值过大的特征。最主要的两种
  - 在``sklearn.feature_selection``里面
  - ``SelectKBest``：选择固定数量的k个特征
  - ``SelectPercentile``：选择固定百分比的特征
- 可视化选择：``get_support`` + ``plt.matshow``



#### 4.5.2 基于模型的特征选择

- 使用一个监督机器学习 模型来判断每个特征的重要性，并且仅保留最重要的特征
- ``from sklearn.feature_selection import SelectFromModel``
- ``SelectFromModel(RandomForestClassifier(... threshold='median'))``，其中threshold选取中位数，就可以选择一半特征



#### 4.5.3 迭代特征选择

- 构建一系列模型，每个模型都适用不同数量的特征。
- 两种基本方法：
  - 开始时没有特征，然后逐个添加特征，直到满足某个终止条件
  - 或从所有特征开始，逐个删除特征，直到满足某个终止条件
- 特殊方法：**递归特征消除（recursive feature elimination, RFE）**
  - 每次递归：舍弃最不重要的特征，并构建新模型
  - 返回条件：留下预设数量的特征
  - ``from sklearn.feature_selection import RFE`



### 4.6 利用专家知识

- 用脑子想特征的现实意义，哪些特征最重要，人为干预





## 模型评估与改进

### 评估指标与评分

### 交叉验证

1. K折交叉验证

- k = 5 or 10, 将数据氛围k份，每一份为一折
- 将第一折作为测试集，其他折作为训练集来训练第一个模型





## 7 处理文本数据

### 7.1 用字符串表示的数据类型

- 四种类型的字符串数据：

  - **分类数据**

    来自固定列表（下拉菜单等）的数据。强烈推荐

  - **可以在语义上映射为类别的自由字符串**

    从文本框中得到的回答属于上述列表中的第二类。最好将这种数据编码为分类变量，你可以利用最常见的条目来选择类别。对于无法归类的回答用“其他”类别来。**需要大量人力**。

  - **结构化字符串数据**（略）

  - **文本数据**

    由短语和句子组成。假设所有文档都适用英语。

    数据集： **语料库**

    每个由单个文本表示的数据点被称为**文档**

### 7.3 将文本数据表示为词袋

- **词袋**：计算语料库中每个单词在每个文本中的出现频次，舍弃结构。

  - **分词（tokenization）**: 将每个文档划分为出现在其中的单词，例如空格和标点划分
  - **构建词表（vocabulary building）**： 收集一个词表，里面包含出现在任意文档中的所有词，并对它们进行编号
  - **编码（encoding）**：对于每个文档，计算词表中每个单词在该文档中的出现频次。

  <img src="/Users/mingxulu/Library/Application Support/typora-user-images/image-20210219163454784.png" alt="image-20210219163454784" style="zoom:67%;" />

  

#### 7.3.1 将词袋应用于玩具数据集

- 在`CountVectorizer`中实现，它是一个transformer

  ```python
  # TBD
  bards_words = None
  from sklearn.feature_extraction.text import CountVectorizer
  vect = CountVectorizer()
  vect.fit(bards_words)
  # Use .vocabulary_ to access the word list
  print("Size:{}".format(len(vect.vocabulary_)))
  # Use transformer to create bags of words
  bag_of_words = vect.transform(bards_words)
  # It's a scipy array. Do not use Numpy array due to potential causing MemoryError
  print("bag_of_words:{}".format(repr(bag_of_words)))
  ```

#### 7.3.2 将词袋应用于电影评论

下面我们将其应用于电影评论情感分析的任务。

- 将训练数据和测试数据加载为字符串列表

  ```python
  # 以训练集为例
  vect = CountVectorizer().fit(text_train)
  X_train = vect.transform(text_train)
  print("X_train:\n{}".format(repr(X_train)))
  ```

- 使用get_feature_names()获得词表的特征

  ```python
  feature_names = vect.get_feature_names()
  print("Number of features: {}".format(len(feature_names)))
  print("First 20 features:\n{}".format(feature_names[:20]))
  print("Features 20010 to 20030:\n{}".format(feature_names[20010:20030]))
  print("Every 2000th feature:\n{}".format(feature_names[::2000]))
  ```

- 在尝试改进特征提取之前，我们先通过实际构建一个分类器来得到性能的量化度量。我们将训练标签保存在y_train 中，训练数据的词袋表示保存在X_train 中，因此我们可以在这个数据上训练一个分类器。对于这样的高维稀疏数据，类似LogisticRegression 的线性模型通常效果最好。

  我们首先使用交叉验证对LogisticRegression 进行评估:

  ```python
  from sklearn.model_selection import cross_val_score
  from sklearn.linear_model import LogisticRegression
  scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=5)
  print("Mean cross-validation accuracy: {:.2f}".format(np.mean(scores)))
  ```

- 我们知道，LogisticRegression 有一个正则化参数C，我们可以通过交叉验证来调节它：

  ```python
  from sklearn.model_selection import GridSearchCV
  param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
  grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
  grid.fit(X_train, y_train)
  
  # 输出最好分数以及最好的C值
  print("Best cross-validation score: {:.2f}".format(grid.best_score_))
  print("Best parameters: ", grid.best_params_)
  ```

- 现在，我们可以在测试集上评估这个参数设置的泛化性能：

  ```python
  X_test = vect.transform(text_test)
  print("{:.2f}".format(grid.score(X_test, y_test)))
  ```

- 下面我们来看一下能否改进单词提取。`CountVectorizer` 使用正则表达式提取词例。默认使用的正则表达式是`\b\w\w+\b`。

  **为了减少不包含信息量的特征（比如数字）**，我们使用`min_df`参数来限制词例至少需要在多少个文档中出现过。

  ```python
  vect = CountVectorizer(min_df=5).fit(text_train)
  X_train = vect.transform(text_train)
  print("X_train with min_df: {}".format(repr(X_train)))
  ```



### 7.4 停用词

- 删除没有信息量的单词还有另一种方法，就是舍弃那些出现次数太多以至于没有信息量的单词。

  有两种主要方法

  1. 使用特定语言的停用词（stopword）列表
  2. 舍弃那些出现过于频繁的单词

  ```python
  from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
  print("Number of stop words: {}".format(len(ENGLISH_STOP_WORDS)))
  print("Every 10th stopword:\n{}".format(list(ENGLISH_STOP_WORDS)[::10]))
  
  # 指定stop_words="english"将使用内置列表。
  # 我们也可以扩展这个列表并传入我们自己的列表。
  vect = CountVectorizer(min_df=5, stop_words="english").fit(text_train)
  X_train = vect.transform(text_train)
  print("X_train with stop words:\n{}".format(repr(X_train)))
  ```



### 7.5 用tf-idf缩放数据

TBD



### 7.7 多个单词的词袋（n元分词）

- 使用词袋表示的主要缺点之一是完全舍弃了单词顺序。

- 使用词袋表示时有一种获取上下文的方法，就是不仅考虑单一词例的计数，而且还考虑相邻的两个或三个词例的计数。

  两个词例被称为二元分词（bigram），三个词例被称为三元分词（trigram），更一般的词例序列被称为n 元分词（n-gram）。

  我们可以通过改变`CountVectorizer` 或`TfidfVectorizer的ngram_range` 参数来改变作为特征的词例范围。`ngram_range` 参数是一个元组，包含要考虑的词例序列的最小长度和最大长度。

  默认情况下，为每个长度最小为1 且最大为1 的词例序列（或者换句话说，刚好1个词例）创建一个特征——单个词例也被称为一元分词（unigram）：

  ```python
  cv = CountVectorizer(ngram_range=(1, 1)).fit(bards_words)
  print("Vocabulary size: {}".format(len(cv.vocabulary_)))
  print("Vocabulary:\n{}".format(cv.get_feature_names()))
  ```

  要想仅查看二元分词（即仅查看由两个相邻词例组成的序列），可以将ngram_range 设置为(2, 2)：

  ```python
  cv = CountVectorizer(ngram_range=(2, 2)).fit(bards_words)
  print("Vocabulary size: {}".format(len(cv.vocabulary_)))
  print("Vocabulary:\n{}".format(cv.get_feature_names()))
  ```

- 添加更长的序列（一直到五元分词）也可能有所帮助，但这会导致特征数量的大大增加，也可能会导致**过拟合**，因为其中包含许多非常具体的特征。

- 我们在IMDb 电影评论数据上尝试使用TfidfVectorizer，并利用网格搜索找出n 元分词的最佳设置：

  ```python
  pipe = make_pipeline(TfidfVectorizer(min_df=5), LogisticRegression())
  # 运行网格搜索需要很长时间，因为网格相对较大，且包含三元分词
  param_grid = {"logisticregression__C": [0.001, 0.01, 0.1, 1, 10, 100],
  "tfidfvectorizer__ngram_range": [(1, 1), (1, 2), (1, 3)]}
  grid = GridSearchCV(pipe, param_grid, cv=5)
  grid.fit(text_train, y_train)
  print("Best cross-validation score: {:.2f}".format(grid.best_score_))
  print("Best parameters:\n{}".format(grid.best_params_))
  ```

- 我们可以将交叉验证精度作为ngram_range 和C 参数的函数并用热图可视化:

  ```python
  # 从网格搜索中提取分数
  scores = grid.cv_results_['mean_test_score'].reshape(-1, 3).T
  # 热图可视化
  heatmap = mglearn.tools.heatmap(
  scores, xlabel="C", ylabel="ngram_range", cmap="viridis", fmt="%.3f",
  xticklabels=param_grid['logisticregression__C'],
  yticklabels=param_grid['tfidfvectorizer__ngram_range'])
  plt.colorbar(heatmap)
  ```

  <img src="/Users/mingxulu/Library/Application Support/typora-user-images/image-20210220142008756.png" alt="image-20210220142008756" style="zoom:67%;" />

  从热图中可以看出，使用二元分词对性能有很大提高，而添加三元分词对精度只有很小贡献。



### 7.8 高级分词、词干提取与词形还原

TBD



### 7.9 主题建模与文档聚类

-  **主题建模 (topic modeling)**, 是描述将每个文档分配给一个或多个主题的任务（通常是无监督的）的概括性术语。

- **隐含狄利克雷分布**

  从直观上来看，LDA 模型试图找出频繁共同出现的单词群组（即主题）。LDA 还要求，每个文档可以被理解为主题子集的“混合”。

  重要的是要理解，机器学习模型所谓的“主题”可能不是我们通常在日常对话中所说的主题，它可能具有语义，也可能没有。

- 我们将LDA应用于**电影评论数据集**， 对于无监督的文本文档模型，通常最好删除非常常见的单词，否则它们可能会支配分析过程。我们将删除至少在15%的文档中出现过的单词，并在删除前15%之后，将词袋模型限定为最常见的10000 个单词：

  ```python
  vect = CountVectorizer(max_features=10000, max_df=.15)
  X = vect.fit_transform(text_train)
  ```

  我们将使用"batch" 学习方法，它比默认方法（"online"）稍慢，但通常会给出更好的结果。我们还将增大max_iter，这样会得到更好的模型:

  ```python
  from sklearn.decomposition import LatentDirichletAllocation
  lda = LatentDirichletAllocation(n_topics=10, learning_method="batch",
  max_iter=25, random_state=0)
  # 我们在一个步骤中构建模型并变换数据
  # 计算变换需要花点时间，二者同时进行可以节省时间
  document_topics = lda.fit_transform(X)
  ```

  LatentDirichletAllocation 有一个`components_ `属性，其中保存了每个单词对每个主题的重要性。`components_ `的大小为(n_topics, n_words)：

  ```python
  lda.components_.shape
  ```

  为了更好地理解不同主题的含义，我们将查看每个主题中最重要的单词。print_topics 函数为这些特征提供了良好的格式：

  ```python
  # 对于每个主题（components_的一行），将特征排序（升序）
  # 用[:, ::-1]将行反转，使排序变为降序
  sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
  # 从向量器中获取特征名称
  feature_names = np.array(vect.get_feature_names())
  
  # 打印出前10个主题：
  mglearn.tools.print_topics(topics=range(10), feature_names=feature_names,
  sorting=sorting, topics_per_chunk=5, n_words=10)
  ```

  





  