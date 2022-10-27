#!/usr/bin/env python
# coding: utf-8

# # Домашнее задание  
# ## Бакуева Дженнет 

# Для начала работы нам необходимо импортировать библиотеки **pandas** и **numpy** для работы с данными.

# In[1]:


import pandas as pd
import numpy as np


# Далее выгружаем датасет для дальнейшей работы. Данные были взяты с сайта kaggle.com.  
# ## Описание данных:  
# В данных представлен большой массив информации, предназначенный для анализа личности потребителя и определения характеристик "идеального" покупателя.  
# Переменные внутри датасета можно разделить на 4 подгруппы: социодемографические показатели, описание продукта, описание маркетинговой компании, описание места. Всего в данных имеется 28 колон с переменными.  
# ### Люди:  
# **ID** - уникальный идентификатор клиента  
# **Year_Birth** - год рождения клиента  
# **Education** - уровень образования клиента  
# **Marital_Status** - семейное положение клиента  
# **Income** - годовой доход семьи клиента  
# **Kidhome** - rоличество детей в семье клиента  
# **Teenhome** - количество подростков в семье клиента  
# **Dt_Customer** - дата регистрации клиента в компании  
# **Recency** - количество дней с момента последней покупки клиента  
# **Complain** - бинарная переменная наличия жалоб, где переменная принимает *значение 1*, если клиент жаловался за последние 2 года, и *значение 0* в противном случае  
# ### Продукт:  
# **MntWines** - сумма, потраченная на вино за последние 2 года  
# **MntFruits** - сумма, потраченная на фрукты за последние 2 года  
# **MntMeatProducts** - сумма, потраченная на мясо за последние 2 года  
# **MntFishProducts** - сумма, потраченная на рыбу за последние 2 года  
# **MntSweetProducts** - сумма, потраченная на сладости за последние 2 года  
# **MntGoldProds** - сумма, потраченная на золото за последние 2 года  
# ### Продвижение:
# **NumDealsPurchases** - количество покупок со скидкой  
# **AcceptedCmp1** - бинарная переменная, где переменная принимает *значение 1*, если клиент принял предложение в 1-й кампании, и *значение 0* в противном случае  
# **AcceptedCmp2** - бинарная переменная, где переменная принимает *значение 1*, если клиент принял предложение в 2-й кампании, и *значение 0* в противном случае  
# **AcceptedCmp3** - бинарная переменная, где переменная принимает *значение 1*, если клиент принял предложение в 3-й кампании, и *значение 0* в противном случае  
# **AcceptedCmp4** - бинарная переменная, где переменная принимает *значение 1*, если клиент принял предложение в 4-й кампании, и *значение 0* в противном случае  
# **AcceptedCmp5** - бинарная переменная, где переменная принимает *значение 1*, если клиент принял предложение в 5-й кампании, и *значение 0* в противном случае  
# **Responce** - бинарная переменная, где переменная принимает *значение 1*, если клиент принял предложение в последней кампании,и *значение 0* в противном случае  
# ### Место:  
# **NumWebPurchases** - количество покупок, совершенных через веб-сайт компании  
# **NumCatalogPurchases** - количество покупок, сделанных с использованием каталога  
# **NumStorePurchases** - количество покупок, совершенных непосредственно в магазинах  
# **NumWebVisitsMonth** - количество посещений веб-сайта компании за последний месяц  
# ## Цели:  
# Хотелось бы отметить, что при выборе данных мною были выделены некоторые цели создания данного исследования, а именно определение зависимости количества покупок и размера потраченных денег от социодемографических показателей клиентов. В дальнейшем все манипуляции с данными будут исходить из цели исследования. Предполагается объединить стоимость всех покупок и количество всех покупок и использовать их в качестве зависимых переменных для исследования.

# In[116]:


df = pd.read_excel('marketing_campaign2.xlsx')


# In[117]:


df.head()


# In[118]:


df.shape


# Мы имеем 2240 строк, то есть у нас есть 2240 наблюдей, по которым мы можем проводить дальнейший анализ. 

# In[119]:


df.info()


# Мы можем видеть, что в наших данных есть пропуски в значении Income. Позже мы будем с ними работать.  
# 

# Отсортируем наши данные по количеству потраченных денег на все категории товаров от меньшего к большему. Для удобства создадим общую перемнную с агрегированными расходами покупателей на все категории товаров. 

# In[120]:


df['Sales'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] +  df['MntGoldProds']


# In[121]:


df.sort_values(by = 'Sales')


# Теперь для дальнейшего удобства удалим ненужные для дальнейшего исследования перемнные:

# In[122]:


del df['MntWines']
del df['MntFruits'] 
del df['MntMeatProducts']
del df['MntFishProducts']
del df['MntSweetProducts']
del df['MntGoldProds']
del df['Dt_Customer'] 
del df['Z_Revenue']
del df['Z_CostContact']


# In[123]:


df.head()


# Создадим переменную возраста *Age* для удобства: 

# In[124]:


df['Age'] = 2022 - df['Year_Birth']


# In[125]:


del df['Year_Birth']


# In[126]:


df['Purchases'] = df['NumWebPurchases'] + df['NumCatalogPurchases'] + df['NumStorePurchases'] + df['NumWebVisitsMonth'] + df['NumDealsPurchases']


# In[127]:


del df['NumWebPurchases']
del df['NumCatalogPurchases']
del df['NumStorePurchases']
del df['NumWebVisitsMonth']
del df['NumDealsPurchases'] 


# In[128]:


df.head()


# Отсортируем данные не только по сумме продаж, но и по их количеству:

# In[131]:


df.sort_values(by = ['Purchases', 'Sales'])


# В данных мы имеем две перемнные, которые определяют наличие детей и подростков в домохозяйстве. Так как для исследования важно само наличие детей и подростков в домохозяйстве, то я переделяю данные переменные *Kidhome* и *Teenhome* в бинарные, где 1 - в семье есть хотя бы 1 ребенок/подросток, и 0 в противном случае.  
# Для этого воспользуемся анонимной функцией lambda

# In[133]:


df['Kidhome'] = df['Kidhome'].apply(lambda x: 0 if x == 0 else 1)


# In[134]:


df['Teenhome'] = df['Teenhome'].apply(lambda x: 0 if x == 0 else 1)


# Для работы с переменными Education и Marital_Status необходимо перевести их в численное значение. Для этого создадим из них категориальные переменные с помощью функции pandas

# In[136]:


df['Education'] = pd.Categorical(df['Education'])


# In[146]:


df.groupby('Education').size()


# Соответсвие значений категориальной переменной образование (так как функция groupby выводит значения в порядке возрастания индекса)  
# Education == 0 (2n Cycle)  
# Education == 1 (Basic)  
# Education == 2 (Graduation)  
# Education == 3 (Master)  
# Education == 4 (PhD)  

# In[147]:


df['Marital_Status'] = pd.Categorical(df['Marital_Status'])


# In[148]:


df.groupby('Marital_Status').size()


# Соответсвие значений категориальной переменной семейного положения (так как функция groupby выводит значения в порядке возрастания индекса)  
# Marital_Status == 0 (Absurd)  
# Marital_Status == 1 (Alone)  
# Marital_Status == 2 (Divorced)  
# Marital_Status == 3 (Married)  
# Marital_Status == 4 (Single)  
# Marital_Status == 5 (Together)  
# Marital_Status == 6 (Widow)  
# Marital_Status == 7 (YOLO)

# Мы имеем 6 бинарных переменных, которые показывают, принял ли клиент предложение с 1 по 6 маркетинговые компании.  
# Для исследования важна информация, принял ли клиент предложение или нет (мы не имеем подробной информации о проводимых маркетинговых кампаниях, поэтому нам важен лишь факт, подействовала ли промо акция на действия клиента или нет).  
# Эта переменная будет показывать, можно ли воздействовать на потенциального клиента с помощью маркетинговых кампаний. Если это возможно, то можно назвать такого клиента более "идеальным" с точки зрения компании.  
# Для того, чтобы создать описанную выше переменную, я сложу все переменные продвижения и далее с помощью ананимной функции создам новую бинарную переменную **Easy_promo**, принимающую значение 1, если клиент хотя бы раз принял предложение кампании, и 0 в противном случае.

# In[ ]:


df['Easy_promo'] = df['AcceptedCmp1'] + df['AcceptedCmp2'] + df['AcceptedCmp3'] + df['AcceptedCmp4'] + df['AcceptedCmp5'] + df['Response']


# In[422]:


def Easy_promo_bi(Easy_promo):
    if Easy_promo == 0:
        return Easy_promo_bi == 0
    else:
        return Easy_promo_bi == 1
df['Easy_promo'] = df['Easy_promo'].apply(Easy_promo_bi)


# In[423]:


df.head()


# In[154]:


del df['AcceptedCmp1']
del df['AcceptedCmp2']
del df['AcceptedCmp3']
del df['AcceptedCmp4']
del df['AcceptedCmp5']
del df['Response']


# In[198]:


print(np.max(df['Purchases']))


# In[200]:


def Purchases_category(Purchases):
    if Purchases <= 15:
        return 'Low'
    elif Purchases >= 30:
        return 'High'
    else:
        return 'Normal'


# Описание значений категорий количества покупок:  
# **Low** - реже, чем раз в 2 месяца  
# **Norm** - приблизительно 1 раз в месяц  
# **High** - чаще, чем 1 раз в месяц 

# In[201]:


df['Purchases'].apply(Purchases_category)


# In[202]:


df['Purchases_Category'] = df['Purchases'].apply(Purchases_category)


# In[203]:


df['Purchases_Category'] = pd.Categorical(df['Purchases_Category'])


# Последним преобразованием в датасете будет добавление переменной логарифма доходов покупателей, чтобы при анализе использовать нормализованные значения, а не абсолютные. Для этого воспользуемся функцией log библиотеки numpy. Будет использоваться натуральный логарифм.

# In[157]:


df['log_Income'] = np.log(df['Income'])


# In[379]:


df['Income'].hist(bins=100, color='blue')


# In[378]:


df['log_Income'].hist(bins=100, color='blue')


# На гистограммах можно заметить, что распределение переменной логарифма доходов больше подходит для Гауссовского распределения. При построении логарифмических и полулогарифмических регрессий коэффициенты будут показывать влиение не в абсолютном значении, а в процентном, что более показательно. 

# In[188]:


df.head()


# Для наглядности посмотрим на простую визуализацию распределения частот значений описательных переменных.

# 1. Распределение переменной Образования 

# In[377]:


df['Education'].value_counts().plot(kind='bar', color='blue')


# In[446]:


df['Education'].describe()


# 2. Распределение переменной Семейного положения 

# In[376]:


df['Marital_Status'].value_counts().plot(kind='bar', color='blue')


# In[447]:


df['Marital_Status'].describe()


# 3.1. Распределение переменной Продаж 

# In[374]:


df['Sales'].hist(color='blue')


# In[449]:


df['Sales'].describe()


# 3.2. Распределение переменной Логарифма Продаж 

# In[375]:


np.log(df['Sales']).hist(color='blue')


# ## Дескриптивная статистика:

# In[215]:


df.describe()


# Ранее мы выяснили, что в нашем датасете имеются пропущенные значения в переменной **Income** и переменной **log_Income** соответсвенно. Посмотрим, сколько у нас пропущенных значений:

# In[224]:


df.isna().sum()


# Пропущенных значений всего 24 по обеим переменным. Теперь удалим наши пропущенные значения

# Чтобы не удалять значения из исходного датафрейма, создадим новый датафрейм **df_clean**

# In[424]:


df_clean = df.dropna(subset=['Income'])


# In[263]:


df_clean.shape


# Теперь мы имеем очищенный от пропущенных переменных датафрейм df_clean

# # Выбросы:  
# Определим, есть ли в наших переменных выбросы. Мы имеем 2 переменные, в которых могут быть выбросы - **Income** и **Sales**. Для определения этих выбросов я импортирую дополнительную библиотеку **scipi.stats**, создам колонки с z-оценками по двум переменным и выведу те значения z-оценок, которые отличаются на 3 среднеквадратичных откланения. 

# In[230]:


import scipy.stats


# 1. Переменная Дохода

# In[233]:


df_clean['z-score_inc']=scipy.stats.zscore(df_clean['Income'])


# In[237]:


df_clean.head()


# 2. Переменная Продаж

# In[238]:


df_clean['z-score_sales']=scipy.stats.zscore(df_clean['Sales'])
df_clean.head()


# In[265]:


df_clean[df_clean['z-score_sales'] < -3]
df_clean[df_clean['z-score_inc'] < -3]


# Можно видеть, что у нас нет выбросов по нижней границе, поэтому рассмотрим верхниии границы

# In[258]:


df_clean['z-score_sales'][df_clean['z-score_sales'] > 3].info()


# По нижней границе мы имеем 5 выбросов для переменной Sales

# In[259]:


df_clean['z-score_inc'][df_clean['z-score_inc'] > 3].info()


# По нижней границе мы имеем 8 выбросов для переменной Income

# Далее воспользуемся функцией вывода среднеквадратичного отклонения. 

# In[266]:


mean = df_clean['Income'].mean()
std_inc = df_clean['Income'].std()
std_inc_bottom = mean - 3 * std_inc
std_inc_top = mean + 3 * std_inc
print(std_inc_bottom, std_inc_top)


# В нижней границе нет отклонений, так как Дохода ниже нуля в данных не имеется. 

# In[267]:


df_clean[df_clean['Income'] > std_inc_top].shape


# Получаем те же 8 отклонений, что и ранее получили с помощью z-оценки. Повторим то же самое со среднеквадратичным отклонением переменнгой Sales 

# In[268]:


mean = df_clean['Sales'].mean()
std_sales = df_clean['Sales'].std()
std_sales_bottom = mean - 3 * std_sales
std_sales_top = mean + 3 * std_sales
print(std_sales_bottom, std_sales_top)


# В нижней границе нет отклонений, так как Продаж ниже нуля в данных не имеется. 

# In[269]:


df_clean[df_clean['Sales'] > std_sales_top].shape


# Получаем те же 5 отклонений, что и ранее получили с помощью z-оценки.

# Проверим влияние этих выбросов на меры центрального значения. Для этого создадим датафреймы без выбросов и определим их медианное и среднее значения:

# In[274]:


df_clean_outliers_s = df_clean[df_clean['Sales'] <= std_sales_top]


# In[275]:


df_clean_outliers_i = df_clean[df_clean['Income'] <= std_inc_top]


# In[280]:


print('Mean income with outliers =', df_clean['Income'].mean())
print('Mean income without outliers =', df_clean_outliers_i['Income'].mean())
print('Mean sales with outliers =', df_clean['Sales'].mean())
print('Mean sales without outliers =', df_clean_outliers_s['Sales'].mean())


# In[282]:


print('Median income with outliers =', df_clean['Income'].median())
print('Median income without outliers =', df_clean_outliers_i['Income'].median())
print('Median sales with outliers =', df_clean['Sales'].median())
print('Median sales without outliers =', df_clean_outliers_s['Sales'].median())


# Так как выбросы не имеют критического влияния на меры центрального значения, то продолжим работать с датафреймом **df_clean** с выбросами.

# In[321]:


df_clean['Income'].describe()


# Добавим дополнительную категориальную переменную уровня дохода, где уровень дохода делится на высокий  
# "High income" (выше 40 000),  
# "Middle income" (больше или равно 15 000 и меньше 40 000) и  
# "Low incime" (меньше 15 000) 

# In[346]:


def Income_cat(Income):
    if Income < 15000:
        return 'Low income'
    if Income >= 15000 and Income < 40000:
        return 'Middle income'
    if Income >= 40000:
        return 'High income'
    else:
        return Income


# In[347]:


df_clean['Income_cat'] = df_clean['Income'].apply(Income_cat)


# In[348]:


df_clean.head()


# # Корреляция:

# В нашем датафрейме имеется небольшое количество количественных переменных: **Income, log_Income, Age, Sales, Puschases** (потому что изначальная логика исследования строилась на построении регрессионных моделей). Так как все остальные переменные являются либо категорииальными, либо бинарными, то строить корреляцию можно только по 5 количественным переменным, перечисленным выше.

# **Построим зависимость Продаж от Доходов:**

# In[453]:


df_clean['Sales'].corr(df_clean['Income'])


# In[455]:


df_clean.plot('Income', 'Sales', kind='scatter', color='blue', figsize = (10,5))


# Корреляция между данными переменными 66%. Так же на графике рассеивания можно увидеть четкую "линию" зависимости Продаж от Дохода

# **Аналогично проведем корреляцию Продаж и логарифма Доходов**

# In[451]:


df_clean['Sales'].corr(df_clean['log_Income'])


# In[456]:


df_clean.plot('log_Income', 'Sales', kind='scatter', color='blue', figsize = (10,5))


# Здесь так же прослеживается зависимость - чем больше Доход, тем выше Продажи в абсолютном значении

# **Построим зависимость Продаж и Возраста покупателя**

# In[291]:


df_clean['Sales'].corr(df_clean['Age'])


# Низкий уровень корреляции 11%

# In[371]:


df_clean.plot('Sales', 'Age', kind='scatter',color='blue')


# На графике есть едва заметная зависимость: чем больше возраст, тем меньше Продажи

# **Построим зависимость Количества Продаж от дохода**

# In[285]:


df_clean['Purchases'].corr(df_clean['Income'])


# Корреляция 42%, что можно назвать умеренным.

# In[372]:


df_clean.plot('Purchases', 'Income', kind='scatter', color='blue')


# **Построим зависимость Количества Продаж от Возраста**

# In[292]:


df_clean['Purchases'].corr(df_clean['Age'])


# Аналогично Sales Количество продаж практически не зависит от Вощраста клиента

# In[373]:


df_clean.plot('Purchases', 'Age', kind='scatter', color='blue')


# На гистограмме рассеивания так же не заметна зависимость

# # Визуализация:  
# Для оформления визуализации воспользуемся дополнительной биьлиотекой **matplotlib.pyplot**

# In[298]:


import matplotlib.pyplot as plt


# **Построим bar chart с Уровнем образования и категориальной перемнной Количества Продаж**

# In[442]:


df_clean.groupby('Education')['Purchases_Category'].value_counts().unstack().plot(kind='bar', stacked = True)


# Можно заметить, что люди с высшим образованием дулают покупки сильно чаще, чем люди с базовым образованием

# **Аналогично построим график Семейного положения и категориальной перемнной Количества Продаж**

# In[443]:


df_clean.groupby('Marital_Status')['Purchases_Category'].value_counts().unstack().plot(kind='bar', stacked = True)


# По этой гистограмме заметно, что люди в отношениях склонны к высокой частоте покупок, чем одинокие.

# **Построим график Наличия детей и категориальной перемнной Количества Продаж**

# In[444]:


df_clean.groupby('Kidhome')['Purchases_Category'].value_counts().unstack().plot(kind='bar', stacked = True)


# Изначально была гипотеза, что люди с детьми чаще производят покупки, чем без детей, однако из этого графика можно заметить, что клиенты без детей производят покупки чаще

# **Построим график Наличия подростков и категориальной перемнной Количества Продаж**

# In[445]:


df_clean.groupby('Teenhome')['Purchases_Category'].value_counts().unstack().plot(kind='bar', stacked = True)


# **Построим графики зависимости Продаж от Образования, Семейного положения и Уровня дохода. Для наглядности приведем аналогичные графики с частотой произведения покупок (Purchases).**

# In[325]:


df_ed_sales = df_clean.groupby('Education')['Sales'].mean()


# In[352]:


df_ed_purchases = df_clean.groupby('Education')['Purchases'].mean()


# In[356]:


df_inc_sales = df_clean.groupby('Income_cat')['Sales'].mean()
df_inc_purchases = df_clean.groupby('Income_cat')['Purchases'].mean()


# In[362]:


df_fam_sales = df_clean.groupby('Marital_Status')['Sales'].mean()
df_fam_purchases = df_clean.groupby('Marital_Status')['Purchases'].mean()


# In[458]:


fig, ax = plt.subplots(2, 3, figsize=(20, 10))
ax[1][0].bar(df_ed_purchases.index, df_ed_purchases, color='red')
ax[0][0].bar(df_ed_sales.index, df_ed_sales, color='pink')
ax[0][1].bar(df_inc_sales.index, df_inc_sales, color='pink')
ax[0][2].bar(df_fam_sales.index, df_fam_sales, color='pink')
ax[1][1].bar(df_inc_purchases.index, df_inc_purchases, color='red')
ax[1][2].bar(df_fam_purchases.index, df_fam_purchases, color='red')


# **Построим зависимость Продаж и частоты покупок от Количества дней с последней покупки:**

# In[380]:


df_last_sales = df_clean.groupby('Recency')['Sales'].mean()
df_last_purchases = df_clean.groupby('Recency')['Purchases'].mean()


# In[404]:


fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(df_last_purchases.index, df_last_purchases, color='red')


# In[402]:


fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(df_last_sales.index, df_last_sales, color='pink')


# # Введение в регрессионный анализ:  
# Далее я приведу примеры регрессионных моделей, которые прпедполагались при обработке исходных данных.

# Для начала импортируем необходимые библиотеки

# In[428]:


import seaborn as sns
import statsmodels.api as sm


# Проверим наши переменные на мультиколлениарность (взаимозависимость описывающих переменных), чтобы убедиться, что наша модель будет эффективной.

# In[430]:


plt.figure(figsize =(20,20))
p=sns.heatmap(df_clean.corr(), annot=True, cmap='RdYlGn', vmin=-1, vmax=1 )


# Можем видеть умеренную корреляцию между переменными

# In[432]:


X = df_clean['Sales']
y = df_clean['Age']

res = scipy.stats.linregress(X, y)
print (res)


# p_value показывает, что модель статистически значима, однако уже видно, что значение r_sq будет низким, то есть модель плохо объясняет дисперсию целеваой переменной

# In[434]:


r_sq = res.rvalue ** 2
r_sq


# In[436]:


X_1 = df_clean['Sales']
y = df_clean['Age']
X_1_cons = sm.add_constant(X_1)

model = sm.OLS(y, X_1_cons)
res_1 = model.fit()
print(res_1.summary())


# В данном домашнем задании не требуется приводить регрессии, однако, несмотря на это многие манипуляции с исходными данными были рассчитаны на дальнейшее использование в регрессиях:  
# 1. Использование зависимые переменные Sales и Purchases для определения реливантных описательных факторов (возраст, логарифм дохода, семейное положение, уровень образования). 
# 2. Помимо этого предполагалось построить модель бинарного выбора с зависимой переменной Easy_promo для определения тех характеристик клиента, которые влияют на восприятие клиентами промоакций и маркетинговыз кампаний. 

# In[ ]:




