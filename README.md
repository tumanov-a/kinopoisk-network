За основу для построения и анализа графа были взяты фильмы из топ-250 Кинопоиска, к каждому из фильмов представленных в данном топе были извлечены похожие фильмы. Вершиной является фильм, ребром является связь между текущим фильмом и фильмом, находящимся в похожих фильмах к данному.

В ноутбуке main_analysis.ipynb проведено извлечение данных с помощью API, первоначальный анализ графа, рассмотрены алгоритмы кластеризации, а также проведен анализ centrality значений вершин.

В utils.py находятся вспомогательные функции, использовавшиеся для анализа графа.