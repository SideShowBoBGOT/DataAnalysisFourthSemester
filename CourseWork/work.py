#!/usr/bin/env python
# coding: utf-8

# # 3 Робота з даними

# ## 3.1 Опис обраних даних

# ### Імпортуємо пакет pandas, завантажимо дані в датафрейм, виводимо інформацію про нього. Для вирішення поставленої задачі було обрано "Cardiovascular Disease" датасет. Він складається з 70 000 пацієнтів. Даний датасет містить в собі такі колонки, як-от: "age" - вік, "height" - зріст, "weight" - вага, "gender" - стать, "ap_hi" - систолічний кров’яний тиск, "ap_lo" - діастолічний артеріальний тиск, "cholesterol" - холестерол, "gluc" - глюкоза, "smoke" - чи курить пацієнт, "alco" - чи вживає алкоголь, "active" - фізична активність, "cardio" - присутність захворювання. Додатково імпортуємо модулі matplotlib.pyplot та seaborn для відображення графіків.

# In[46]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('data/cardio_train.csv', sep=';')
df


# *Завантажений датафрейм*

# ### Видалимо колонку id за допомогою методу pandas.Dataframe.drop, передавши в нього список з єдиним елементом id та параметром за замовчуванням inplace=True, щоб зміна відбулася в самому датафреймі, а не видало новий. 

# In[ ]:


df.drop(['id'], axis=1, inplace=True)


# *Видалення колонки "id"*

# In[47]:


df.isna().any()


# *Колонки, де всі значення ініціалізовані*

# ## 3.2 Вибір ознак для аналізу

# ### Побудуємо матрицю кореляції та визначимо, наскільки кожен фактор впливає на якість води. Для цього імпортуємо модуль seaborn та застосуємо функцію heatmap, де передаємо йому вище зазначену матрицю, використавши pandas.Dataframe.corr метод датафрейму. 

# In[13]:


import seaborn as sns
fig, axis = plt.subplots(figsize=(10, 6))
axis.set_title('Кореляція між факторами')
sns.heatmap(df.corr(), ax=axis, annot=True)


# *Матриця кореляцій*

# ### Бачимо, що найбільше на виникнення серцево-судинних захворювань впливає вік, вага, холестерол. Доволі цікавим фактами є кореляція між віком та холестеролом, статтю та курінням, статтю та алкоголем, холестеролом та глюкозою. Залежність між статтю та вагою і висотою доволі очевидна, що пояснюється звичайною різницею у фізичних показниках між чоловіком та жінокю. Зобразимо статистику факторів та виникненням захворювання, розділену за статтю. За допомогою функції seaborn.FacetGrid згрупуємо значення та за допомогою seaborn.histplot зобразимо їх у вигляді гістограми. Побудуємо статистику за віком.

# In[48]:


def plot_stat(factor: str):
    g = sns.FacetGrid(df[['gender', 'cardio'] + [factor]], col='gender', hue='cardio')
    g.map(sns.histplot, factor)
    g.add_legend()
plot_stat('age')


# *Статистика захворювання від віку між чоловіками та жінками*

# ### Зобразимо статистику за вагою.

# In[15]:


plot_stat('weight')


# *Статистика захворювання від ваги між чоловіками та жінками*

# ### Зобразимо статистику за холестеролом.

# In[16]:


plot_stat('cholesterol')


# *Статистика захворювання від холестеролу між чоловіками та жінками*

# ## 3.3 Поділ даних

# ### Для збільшення якості моделі залишимо лише чотири колонки: age, weight, cholesterol, cardio.

# In[49]:


df = df[['age', 'weight', 'cholesterol', 'cardio']]
df


# *Відфільтрований датафрейм*

# ### Ділимо дані на тренувальні та тестові для подальшох роботи. Імпортуємо модуль sklearn.model_selection та застосуємо функцію train_test_split. Розділимо набір даних на 80% навчальних та 20% тестових.

# In[50]:


from sklearn.model_selection import train_test_split
x = df.iloc[:, :-1]
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, random_state=0)


# *Поділ інформації на інформацію та результат*

# ### Зберігатимемо результати тестування моделей у списку results

# In[ ]:


results = []


# *Список результатів*

# # 4 Інтелектуальний аналіз даних

# ## 4.1 Обґрунтування вибору методів інтелектуального аналізу даних

# ### Було обрано чотири методи K-Nearest Neighbors, Logistic Regression, Random Forest, SVM.

# ### K-Nearest Neighbors (KNN) — це простий, але ефективний алгоритм машинного навчання, який використовується для завдань класифікації та регресії. Я обрав, щоб порівняти наскільки він швидший за інші алгоритми, оскільки він не робить жодних припущень щодо розподілу даних.

# ### До того ж однією з сильних сторін алгоритму KNN є те, що він легко інтерпретується. Прогнози можна пояснити, просто подивившись на K найближчих сусідів даної точки даних. Крім того, KNN може обробляти нелінійні межі рішень і може використовуватися як для бінарних, так і для багатокласових задач класифікації. Тому він чудово підійде для прогнозування хвороб, де треба враховувати багато факторів.

# ### Однак, якщо датасети великі, то KNN може бути дорогим з обчислювальної точки зору. Крім того, продуктивність KNN значною мірою залежить від значення K, яке потрібно ретельно вибирати для досягнення оптимальних результатів.

# ### Logistic Regression – це статистичний метод, який використовується для прогнозування ймовірності події. Оскільки її залежна змінна є двійковою, то це дуже підходить для виявлення хвороби: вона або є, або її нема.

# ### Logistic Regression має кілька переваг перед іншими алгоритмами класифікації. По-перше, її легко реалізувати та інтерпретувати. По-друге, вона може обробляти як категоричні, так і безперервні незалежні змінні. Тобто можна обробляти як дискретні дані, які лекго визначити: вага, зріст - так і виявлення в організмі сполук, які складно визначити точно, і де робляться припущення про інтервали: рівень глюкози, рівень холестиролу, вітамінів в організмі тощо. По-третє, він може надати ймовірність події, що станеться, що корисно під час медичної діагностики.

# ### Random Forest — це популярний алгоритм машинного навчання, який використовується для завдань класифікації та регресії. Це метод ансамблю, який поєднує передбачення кількох дерев рішень для отримання більш точних результатів.

# ### Я його обрав, бо є стійким до шуму. До того ж цікаво, як він буде передбачувати наявність хвороби в пацієнта. Варто зауважити, що Random Forest може бути дорогим з точки зору обчислень, особливо при роботі з великими наборами даних і багатьма деревами. Також може бути важко інтерпретувати результати випадкового лісу, оскільки процес прийняття рішення розподіляється між кількома деревами.

# ### SVM - це популярний і потужний клас алгоритмів керованого навчання, які використовуються для завдань класифікації та регресії. SVM особливо добре підходять для проблем, де існують складні межі між класами. У медичній сфері це часто використовується для прогнузування пацієту декількох хвороб, де окреме захворювання - це клас, де варто мати чіткі межі для поставлення правильного діагнозу.

# ### Основна ідея SVM — знайти гіперплощину, яка найкраще розділяє дані на різні класи. Гіперплощина — це лінійна межа рішення, яка розділяє два класи в просторі ознак.

# ### Однією з сильних сторін SVM є те, що вони нечутливі до викидів, оскільки на гіперплощину впливають лише найближчі до неї точки. Це робить SVM дуже стійкими до зашумлених даних, де зазвичай пацієнтів дуже багато, а тому ця особливість є важливою.

# ### Однак SVM мають деякі обмеження. Навчання їх може бути дорогим з обчислювальної точки зору, особливо з великими наборами даних або просторами функцій великої розмірності. Крім того, вибір правильної функції ядра може бути складним, а продуктивність SVM залежить від вибору гіперпараметрів.

# ## 4.2 Аналіз отриманих результатів для методу K-Nearest Neighbors

# ### Для виконання роботи методу KNN імпортуємо sklearn.neighbors.KNeighborsClassifier та sklearn.model_selection.GridSearchCV. 

# In[19]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


# *Імпортування модулів*

# ### Визначимо, які варіанти параметрів найкраще вирішують дану задачу.

# In[20]:


classificator = KNeighborsClassifier()
params = {'n_neighbors': range(1, 60)}
grid_search = GridSearchCV(classificator, params, cv=10, verbose=1)
grid_search.fit(x_train, y_train)
knn = grid_search.best_estimator_
knn


# *Визначення найкращого параметра*

# ### Натренуємо модель з найкращим параметром.

# In[21]:


knn.fit(x_train, y_train)


# *Тренування моделі K-Nearest Neighbors*

# ### Визначимо точність моделі на тренувальних та тестових даних

# In[51]:


train_score = round(knn.score(x_train, y_train), 5)
test_score = round(knn.score(x_test, y_test), 5)
results.append({'method': 'knn', 'score': train_score, 'type': 'train'})
results.append({'method': 'knn', 'score': test_score, 'type': 'test'})
print(f'Train accuracy: {train_score}')
print(f'Test accuracy: {test_score}')


# *Точність моделі K-Nearest Neighbors*

# ### Визначимо продуктивність роботи моделі на прикладі матриці невідповідностей. Для цього застосуємо sklearn.metrics.plot_confusion_matrix.

# In[23]:


from sklearn.metrics import confusion_matrix
def conf_mat(model, x_test, y_test):
    y_predicted = model.predict(x_test)
    cm = confusion_matrix(y_test, y_predicted)
    plt.figure(figsize = (8,5))
    sns.heatmap(cm, annot=True, fmt=".1f")
    plt.xlabel('Predicted')


# *Імпортування модуля*

# ### Матриця має два рядки та дві колонки: перший ряд і перша колонка - це істинно позитивні значення, тобто людина здорова і модель не визначила захворювання; першмй ряд і друга колонка - хибно позитивні, тобто людина здорова, а модель сказала, що є захворювання; другий ряд і перша колонка - хибно негативні, тобто людина хвора, а модель сказала, що здорова; другий ряд і друга колонка - істинно негативні, тобто людина хвора і модель сказала, що хвора.

# In[24]:


conf_mat(knn, x_test, y_test)


# *Матриця невідповідностей для K-Nearest Neighbors*

# ### Побудуємо графік ROC( Receiver Operating Characteristic ), що є графіком істинно позитивної відносної частоти проти хибно позитивної частоти. Це показує компроміс між чутливістю та специфічністю. Для цього імпортуємо sklearn.metrics.roc_curve та sklearn.metrics.roc_auc_score. До того ж визначимо AUC( Area Under the ROC Curve ), що є мірою того, наскільки добре модель може розрізняти позитивні та негативні рузультати. Він коливається від 0 до 1, де 1 є найкращим класифікатором, а 0,5 – випадковим класифікатором. AUC корисний під час порівняння продуктивності різних класифікаторів на одному наборі даних, бо даж єдине число, яке підсумовує загальну продуктивність.

# In[25]:


from sklearn.metrics import roc_curve, roc_auc_score
def roc(model, x_test, y_test):
    y_pred_proba = model.predict_proba(x_test)[::,1]
    fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="Area = "+str(auc)+')')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc=4)
    plt.show()


# *Імпортування модуля та визначення функції roc*

# ### Побудуємо ROC для K-Nearest Neighbors.

# In[26]:


roc(knn, x_test, y_test)


# *Графік ROC для K-Nearest Neighbors*

# ## 4.3 Аналіз отриманих результатів для методу Logistic Regression

# ### Для виконання роботи методу KNN імпортуємо sklearn.linear_model.LogisticRegression; sklearn.pipeline.Pipeline для побудови пайплайну, щоб у зручному вигляді передавати параметри до декількох функцій; sklearn.preprocessing.StandardScaler для скейлингу даних до проміжку від 0 до 1. Визначимо найкращі параметри моделі, передавши в неї параметри регуляризації.

# In[27]:


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
logisticRegr = LogisticRegression()
pipe = Pipeline(steps=[('sc', sc), ('logisticRegr', logisticRegr)])
c = np.logspace(-4, 4, 60)
penalty = ['l1', 'l2']
params = dict(logisticRegr__C=c, logisticRegr__penalty=penalty)
log_reg = GridSearchCV(pipe, params)
log_reg.fit(x_train, y_train)


# *Тренування моделі Logistic Regression*

# ### Визначимо точність моделі на тренувальних та тестових даних

# In[52]:


train_score = round(log_reg.score(x_train, y_train), 5)
test_score = round(log_reg.score(x_test, y_test), 5)
results.append({'method': 'logress', 'score': train_score, 'type': 'train'})
results.append({'method': 'logress', 'score': test_score, 'type': 'test'})
print(f'Train accuracy: {train_score}')
print(f'Test accuracy: {test_score}')


# *Точність моделі Logistic Regression*

# ### Визначимо продуктивність роботи моделі на прикладі матриці невідповідностей.

# In[29]:


conf_mat(log_reg, x_test, y_test)


# *Матриця невідповідностей для Logistic Regression*

# ### Побудуємо графік ROC для Logistic Regression.

# In[30]:


roc(log_reg, x_test, y_test)


# *Графік ROC для Logistic Regression*

# ## 4.4 Аналіз отриманих результатів для методу Random Forest

# ### Для виконання роботи методу KNN імпортуємо sklearn.ensemble.RandomForestClassifier. Визначимо найкращі параметри для моделі. У випадку Random Forest параметри включають кількість дерев рішень та кількість характеристик, які враховуються кожним деревом під час поділу вузла.використовуються для поділу кожного вузла, отриманого під час навчання. Імпортуємо sklearn.model_selection.RandomizedSearchCV.

# In[31]:


from sklearn.ensemble import RandomForestClassifier
import numpy as np
# Кількість дерев
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 300, num = 60)]
params = {'n_estimators': n_estimators}
rf = RandomForestClassifier()
rf_random = GridSearchCV(rf, param_grid=params, cv=3, n_jobs=5)
rf_random.fit(x_train, y_train)


# *Тренування моделі Random Forest*

# ### Визначимо точність моделі на тренувальних та тестових даних

# In[53]:


train_score = round(rf_random.score(x_train, y_train), 5)
test_score = round(rf_random.score(x_test, y_test), 5)
results.append({'method': 'rf', 'score': train_score, 'type': 'train'})
results.append({'method': 'rf', 'score': test_score, 'type': 'test'})
print(f'Train accuracy: {train_score}')
print(f'Test accuracy: {test_score}')


# *Точність моделі Random Forest*

# ### Визначимо продуктивність роботи моделі на прикладі матриці невідповідностей.

# In[33]:


conf_mat(rf_random, x_test, y_test)


# *Матриця невідповідностей для Random Forest*

# ### Побудуємо графік ROC для Logistic Regression.

# In[34]:


roc(rf_random, x_test, y_test)


# *Графік ROC для Random Forest*

# ## 4.5 Аналіз отриманих результатів для методу Support Vector Machines

# ### Для виконання роботи методу SVM імпортуємо sklearn.svm.SVC. Визначимо найкращі параметри для моделі. Оскільки складність SVM - це O(n_samples^2 * n_features), тобто для якщо факторів 3 і 70 000 зразків, то маємо 1.47e10 ітерацій, що надзвичайно багато. Тому оберемо 1000 випадковиз зразків і отримаємо загальну кільскість ітерацій в 3e6.

# In[37]:


from sklearn.svm import SVC
import numpy as np
df_svm = df.sample(n=500)
svm_x = df_svm.iloc[:, :-1]
svm_y = df_svm.iloc[:, -1]
svm_x_train, svm_x_test, svm_y_train, svm_y_test = train_test_split(svm_x, svm_y, test_size=0.2, train_size=0.8, random_state=0)
c = [1, 5]
params = {'C': c, 'kernel': ['rbf', 'linear']}
svc = SVC(gamma='auto', probability=True)
svc_model = GridSearchCV(svc, param_grid=params, cv=3, n_jobs=5)
svc_model.fit(svm_x_train, svm_y_train)


# *Тренування моделі SVM*

# ### Визначимо точність моделі на тренувальних та тестових даних

# In[54]:


train_score = round(svc_model.score(x_train, y_train), 5)
test_score = round(svc_model.score(x_test, y_test), 5)
results.append({'method': 'svm', 'score': train_score, 'type': 'train'})
results.append({'method': 'svm', 'score': test_score, 'type': 'test'})
print(f'Train accuracy: {train_score}')
print(f'Test accuracy: {test_score}')


# *Точність моделі SVM*

# ### Визначимо продуктивність роботи моделі на прикладі матриці невідповідностей.

# In[39]:


conf_mat(svc_model, svm_x_test, svm_y_test)


# *Матриця невідповідностей для SVM*

# ### Побудуємо графік ROC для SVM.

# In[40]:


roc(svc_model, x_test, y_test)


# *Графік ROC для SVM*

# ## 4.6 Порівняння отриманих результатів методів

# ### Проаналізувавши окремо кожен із методів, проведемо порівнянняданих методів.

# In[55]:


df_score = pd.DataFrame(results, columns=['method','score','type'])
df_score


# *Датафрейм результатів*

# ### Для наочності побудуємо гістограму.

# In[56]:


sns.barplot(x='method', y='score', hue='type', data=df_score)


# *Результати моделей*

# ### З огляду бачимо, що на тестових даних усі методи відпрацювали більш-менш однаково. Однак на тренувальних даних Random Forest показує себе найкраще.
