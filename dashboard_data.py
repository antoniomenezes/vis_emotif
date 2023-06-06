import pandas as pd
import streamlit as st
import mariadb
import sys

DATA_DIR = './data'
dataset_dir = DATA_DIR

mariadb_user = 'set_your_user'
mariadb_pass = 'set_your_password'

GO_EMOTIONS_DIR = dataset_dir
HF_DIR = dataset_dir
ISEAR_DIR = dataset_dir
SAIF_MOHAMMAD_DIR = dataset_dir
SEMEVAL19_DIR = dataset_dir

dict_emotions_colors = {
    "anger":"#ff0000",
    "fear":"#009600",
    "joy":"#ffff54", # before: "#ffe778",
    "love":"#ff00ff",
    "sadness":"#5151ff",
    "surprise":"#59bdff",
    "admiration":"#00b400",
    "amusement":"#af00ff",
    "annoyance":"#ff8c8c",
    "approval":"#77f400",
    "caring":"#ffb3d4",
    "confusion":"#620f9a",
    "curiosity":"#11b3d4",
    "desire":"#f51e7b",
    "disappointment":"#9a4781",
    "disapproval":"#a4b8ff",
    "disgust":"#ff54ff",
    "embarrassment":"#620f3d",
    "excitement":"#ffe854", #ecstasy
    "gratitude":"#f5ddb1",
    "grief":"#2e2e3f", # before: "#0000c8",
    "nervousness":"#6a57a0",
    "neutral":"#cccccc",
    "optimism":"#fee1aa",
    "pride":"#d4551b",
    "realization":"#00ffa6",
    "relief":"#8abdd6",
    "remorse":"#c5aeff",
    "guilt":"#8cc68c", #apprehension
    "shame":"#7a6c93",  #"#392e50", 
    "others":"#7f7f7f"
}

dict_datasets_colors = {
    'CARER': '#1f77b4',
    'GoEmotions': '#aec7e8',
    'ISEAR': '#ff7f0e',
    'SemEval2019 Task3': '#ffbb78',
    'TEC': '#2ca02c'
}

def get_emotion_color(emotion):
    return dict_emotions_colors[emotion]

def get_dataset_color(source):
    return dict_datasets_colors[source]


ge_labels = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']
carer_labels = ['anger','fear','joy','love','sadness','surprise']
isear_labels = ['anger','disgust','fear','guilt','joy','sadness','shame']
se_labels = ['anger','joy','others','sadness']
tec_labels = ['anger','disgust','fear','joy','sadness','surprise']

simple_nrc_labels = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
nrc_labels = ['nrc_anger', 'nrc_disgust', 'nrc_fear', 'nrc_joy', 'nrc_sadness', 'nrc_surprise']
liwc_emotion_labels = ['liwc_affect','liwc_posemo','liwc_negemo','liwc_anx','liwc_anger','liwc_sad']
liwc_other_labels = ['liwc_negate','liwc_social','liwc_cogmech','liwc_percept','liwc_bio','liwc_relativ','liwc_relig','liwc_death']


@st.cache_data
def df_to_plotly(df):
    return {'z': df.values.tolist(),
            'x': df.columns.tolist(),
            'y': df.index.tolist()}


@st.cache_data
def get_datasets_overview():
    dict_overview = {
        'Dataset' : ['CARER', 'GoEmotions', 'ISEAR', 'SemEval 2019 Task 3', 'TEC'],
        'Year' : ['2018', '2020', '1997', '2019', '2012'],
        '#Emotions' : ['6', '28', '7', '4 (*)', '6'],
        'Domain' : ['Tweets', 'Reddit comments', 'Description of situations', 'Dialogs', 'Tweets'],
        'Annotation Task' : ['Automatic with hashtags','Manual','Questionnaire','Collected from a pool of dialogs','Automatic with hashtags']
    }
  
    df_overview = pd.DataFrame(dict_overview)
    return df_overview

@st.cache_data
def get_color(emotion):
    return dict_emotions_colors[emotion]

@st.cache_data
def get_emotions_colors(emotions):
    emotions_colors =[get_color(emotion) for emotion in emotions]
    return emotions_colors

@st.cache_data
def get_source_emotion():
    if 'df_source_emotion' in st.session_state:
        df_source_emotion = st.session_state.df_source_emotion
    else:
        df_source_emotion = pd.read_csv(dataset_dir+'/data_source_emotion.csv', encoding='UTF-8', sep='|')
        df_source_emotion = df_source_emotion[['source','emotion','counts']]
        st.session_state.df_source_emotion = df_source_emotion
    return df_source_emotion

@st.cache_data
def get_source_emotion_length():
    df_emotion_source_length = pd.read_csv(dataset_dir+'/emotion_source_length.csv', encoding='UTF-8', sep='|')
    return df_emotion_source_length

@st.cache_data
def get_source_emotion_stats_linesize(datasetname, min_textsize, max_textsize):
    try:
        conn = mariadb.connect(
            user=mariadb_user,
            password=mariadb_pass,
            host="localhost",
            port=3306,
            database="vis_emotif"
        )

        # Get Cursor
        cur_stats = conn.cursor()

        if datasetname == 'All Datasets':
            cur_stats.execute("SELECT `emotion`,`source`,`length`,`n_sents`,`n_words`,`diversity_ttr`,`flesch_reading_ease`,`n_ents` FROM data_text_emotion_source_stats WHERE length >= ? and length < ?", (min_textsize, max_textsize))
        else:
            cur_stats.execute("SELECT `emotion`,`source`,`length`,`n_sents`,`n_words`,`diversity_ttr`,`flesch_reading_ease`,`n_ents` FROM data_text_emotion_source_stats WHERE source = ? and length >= ? and length < ?", (datasetname, min_textsize, max_textsize))

        columns = cur_stats.description 
        result = [{columns[index][0]:column for index, column in enumerate(value)} for value in cur_stats.fetchall()]

        if result:
            df_result = pd.DataFrame.from_dict(result)
        else:
            df_result = pd.DataFrame(columns=['emotion','source','length','n_sents','n_words','diversity_ttr','flesch_reading_ease','n_ents'])
        conn.close()
        return df_result

    except mariadb.Error as e:
        st.error(f"Error connecting to MariaDB Platform: {e}")
        return None

@st.cache_data
def get_text_dup_max_min_lengths(datasetname, min_textsize, max_textsize):
    try:
        conn = mariadb.connect(
            user=mariadb_user,
            password=mariadb_pass,
            host="localhost",
            port=3306,
            database="vis_emotif"
        )

        # Get Cursor
        cur_dup = conn.cursor()

        if datasetname == 'All Datasets':
            cur_dup.execute("SELECT MIN(`length`) AS `min_length`, MAX(`length`) AS `max_length` " \
                            "FROM data_text_emotion_source_dup " \
                            "WHERE `length` >= ? AND `length` < ?", (min_textsize, max_textsize))
        else:
            cur_dup.execute("SELECT MIN(`length`) AS `min_length`, MAX(`length`) AS `max_length` " \
                            "FROM data_text_emotion_source_dup " \
                            "WHERE `length` >= ? AND `length` < ? AND `source` = ?", (min_textsize, max_textsize, datasetname))

        columns = cur_dup.description 
        result = [{columns[index][0]:column for index, column in enumerate(value)} for value in cur_dup.fetchall()]

        if result:
            df_result = pd.DataFrame.from_dict(result)
        else:
            df_result = pd.DataFrame(columns=['min_length','max_length'])
        conn.close()
        return df_result

    except mariadb.Error as e:
        st.error(f"Error connecting to MariaDB Platform: {e}")
        return None




@st.cache_data
def get_source_text_dup(datasetname, min_textsize, max_textsize, group_by_emotions = False, num_bins = 20):
    try:
        conn = mariadb.connect(
            user=mariadb_user,
            password=mariadb_pass,
            host="localhost",
            port=3306,
            database="vis_emotif"
        )

        df_min_max_lengths = get_text_dup_max_min_lengths(datasetname, min_textsize, max_textsize)
        df_min_max_lengths = df_min_max_lengths.fillna(0)

        if df_min_max_lengths.empty or (df_min_max_lengths.min_length[0] == 0 and df_min_max_lengths.max_length[0] == 0):
            st.warning("No Data Found")
            return None

        try:
            min_length = df_min_max_lengths.min_length[0]
            max_length = df_min_max_lengths.max_length[0]
        except:
            st.warning("No Data Found")
            return None

        # When there's no way to divide per bins
        if min_length == max_length:
            max_length = max_length + (2*num_bins)

        step_length = round((max_length - min_length)/num_bins);
        step_digits = len(str(max_length))

        first_values = [lin for lin in range(min_length, max_length, step_length)]
        first_values[len(first_values)-1] = max_length
    
        bins = [[i, first_values[i], first_values[i+1]] for i in range(0, len(first_values)-1)]

        # Get Cursor
        cur_dup = conn.cursor()

        sql_code_bins = ""
        for bin in bins:
            sql_code_bins = sql_code_bins + "SELECT "+str(bin[0])+" AS bin_id, '"+str(bin[1]).zfill(step_digits)+" - "+str(bin[2]).zfill(step_digits)+"' AS `bin`, "+str(bin[1])+" AS `left`, "+str(bin[2])+" `right` UNION ALL "+chr(10)
        sql_code_bins = sql_code_bins[0:len(sql_code_bins)-11]      

        if datasetname == 'All Datasets':
            sql_code = "SELECT total.`source`, total.`bin_id`, total.`bin`, dup.`n_lines_dup`, total.`n_lines` " \
                "FROM " \
                "( " \
                "   SELECT s.`source`, b.`bin_id`, b.`bin`, COUNT(*) AS n_lines " \
                "   FROM data_text_emotion_source_stats s, " \
                "   ( " \
                "           "+sql_code_bins+"    ) AS b " \
                "   WHERE s.`length` >= "+str(min_textsize)+" AND s.`length` < "+str(max_textsize)+" AND " \
                "           s.`length` BETWEEN b.`left` AND b.`right` " \
                "   GROUP BY s.`source`, b.`bin_id`, b.`bin` " \
                "   ) AS total, " \
                "   ( " \
                "   SELECT d.`source`, b.`bin_id`, b.`bin`, COUNT(*) AS n_lines_dup " \
                "   FROM data_text_emotion_source_dup d, " \
                "   ( " \
                "           "+sql_code_bins+"   ) AS b " \
                "   WHERE d.`length` >= "+str(min_textsize)+" AND d.`length` < "+str(max_textsize)+" AND " \
                "         d.`length` BETWEEN b.`left` AND b.`right` " \
                "   GROUP BY d.`source`, b.`bin_id`, b.`bin` " \
                ") AS dup " \
                "WHERE total.`source` = dup.`source` AND " \
                "       total.`bin` = dup.`bin` "

            cur_dup.execute(sql_code)

        else:
            if group_by_emotions:
                sql_code = "SELECT total.`source`, total.`emotion`, total.`bin_id`, total.`bin`, dup.`n_lines_dup`, total.`n_lines` " \
                    "FROM " \
                    "( " \
                    "   SELECT s.`source`, s.`emotion`, b.`bin_id`, b.`bin`, COUNT(*) AS n_lines " \
                    "   FROM data_text_emotion_source_stats s, " \
                    "   ( " \
                    "           "+sql_code_bins+"    ) AS b " \
                    "   WHERE s.`length` >= "+str(min_textsize)+" AND s.`length` < "+str(max_textsize)+" AND " \
                    "           s.`length` BETWEEN b.`left` AND b.`right` " \
                    "   GROUP BY s.`source`, s.`emotion`, b.`bin_id`, b.`bin` " \
                    "   ) AS total, " \
                    "   ( " \
                    "   SELECT d.`source`, d.`emotion`, b.`bin_id`, b.`bin`, COUNT(*) AS n_lines_dup " \
                    "   FROM data_text_emotion_source_dup d, " \
                    "   ( " \
                    "           "+sql_code_bins+"   ) AS b " \
                    "   WHERE d.`length` >= "+str(min_textsize)+" AND d.`length` < "+str(max_textsize)+" AND " \
                    "         d.`length` BETWEEN b.`left` AND b.`right` " \
                    "   GROUP BY d.`source`, d.`emotion`, b.`bin_id`, b.`bin` " \
                    ") AS dup " \
                    "WHERE total.`source` = dup.`source` AND total.`emotion` = dup.`emotion` AND " \
                    "       total.`bin` = dup.`bin` AND total.`source` = '"+datasetname+"'"
                cur_dup.execute(sql_code)

            else:
                sql_code = "SELECT total.`source`, total.`bin_id`, total.`bin`, dup.`n_lines_dup`, total.`n_lines` " \
                    "FROM " \
                    "( " \
                    "   SELECT s.`source`, b.`bin_id`, b.`bin`, COUNT(*) AS n_lines " \
                    "   FROM data_text_emotion_source_stats s, " \
                    "   ( " \
                    "           "+sql_code_bins+"    ) AS b " \
                    "   WHERE s.`length` >= "+str(min_textsize)+" AND s.`length` < "+str(max_textsize)+" AND " \
                    "           s.`length` BETWEEN b.`left` AND b.`right` " \
                    "   GROUP BY s.`source`, b.`bin_id`, b.`bin` " \
                    "   ) AS total, " \
                    "   ( " \
                    "   SELECT d.`source`, b.`bin_id`, b.`bin`, COUNT(*) AS n_lines_dup " \
                    "   FROM data_text_emotion_source_dup d, " \
                    "   ( " \
                    "           "+sql_code_bins+"   ) AS b " \
                    "   WHERE d.`length` >= "+str(min_textsize)+" AND d.`length` < "+str(max_textsize)+" AND " \
                    "         d.`length` BETWEEN b.`left` AND b.`right` " \
                    "   GROUP BY d.`source`, b.`bin_id`, b.`bin` " \
                    ") AS dup " \
                    "WHERE total.`source` = dup.`source` AND " \
                    "       total.`bin` = dup.`bin` AND total.`source` = '"+datasetname+"'"

                cur_dup.execute(sql_code)


        columns = cur_dup.description 
        result = [{columns[index][0]:column for index, column in enumerate(value)} for value in cur_dup.fetchall()]

        if result:
            df_result = pd.DataFrame.from_dict(result)
        else:
            df_result = pd.DataFrame(columns=['emotion','source','bin_id','bin','n_lines_dup','n_lines'])
        conn.close()

        return df_result

    except mariadb.Error as e:
        st.error(f"Error connecting to MariaDB Platform: {e}")
        return None

@st.cache_data
def get_path_levels():
    df_path_levels = pd.read_csv(dataset_dir+'/source_emotion_path_levels.csv', encoding='UTF-8', sep='|')
    return df_path_levels

@st.cache_data
def get_max_path_nodes(source, emotion, min_counts):
    try:
        conn = mariadb.connect(
            user=mariadb_user,
            password=mariadb_pass,
            host="localhost",
            port=3306,
            database="vis_emotif"
        )

        # Get Cursor
        cur_max_path = conn.cursor()
        cur_max_path.execute("SELECT max(path_length) as max_path_length FROM source_emotion_path_levels_counts WHERE source=? and emotion=? and counts>?", (source,emotion,min_counts))
        result = [value for value in cur_max_path.fetchone()]

        max_path_nodes = result[0]

        conn.close()
        return max_path_nodes

    except mariadb.Error as e:
        st.error(f"Error connecting to MariaDB Platform: {e}")
        return None
          

@st.cache_data
def get_path_levels(source, emotion, min_counts):
    max_path_nodes = get_max_path_nodes(source, emotion, min_counts)

    level_columns = ['level'+str(i) for i in range(max_path_nodes)]
    level_columns_str = str(level_columns).replace('[','').replace(']','').replace("'","")

    try:
        conn = mariadb.connect(
            user=mariadb_user,
            password=mariadb_pass,
            host="localhost",
            port=3306,
            database="vis_emotif"
        )

        cur_paths = conn.cursor()
        cur_paths.execute("SELECT source, emotion, path, counts, "+level_columns_str+" FROM source_emotion_path_levels_counts WHERE source=? and emotion=? and counts>?", (source,emotion,min_counts))
        
        columns = cur_paths.description 
        result = [{columns[index][0]:column for index, column in enumerate(value)} for value in cur_paths.fetchall()]

        df_result = pd.DataFrame.from_dict(result)
        conn.close()
        return df_result

    except mariadb.Error as e:
        st.error(f"Error connecting to MariaDB Platform: {e}")
        sys.exit(1)


@st.cache_data
def get_table_for_pearson(dataset_name, heatmap_relation, table_name):
    try:
        conn = mariadb.connect(
            user=mariadb_user,
            password=mariadb_pass,
            host="localhost",
            port=3306,
            database="vis_emotif"
        )

        if dataset_name == 'CARER':
            columns = carer_labels
        elif dataset_name == 'GoEmotions':
            columns = ge_labels
        elif dataset_name == 'ISEAR':
            columns = isear_labels
        elif dataset_name == 'TEC':
            columns = tec_labels
        elif dataset_name == 'SemEval2019 Task3':
            columns = se_labels

        if heatmap_relation == 'Emotions + emotions (NRC)':
            columns= columns + nrc_labels

        elif heatmap_relation == 'Emotions + emotions (LIWC)':
            columns = columns + liwc_emotion_labels

        elif heatmap_relation == 'Emotions + tags (LIWC)':
            columns = columns + liwc_other_labels

        # Get Cursor
        cur_stats = conn.cursor()

        sql_columns = str(columns).replace("[","").replace("]","").replace("'","")

        sql_code = "SELECT "+sql_columns+" FROM "+table_name

        cur_stats.execute(sql_code)

        columns = cur_stats.description 
        result = [{columns[index][0]:column for index, column in enumerate(value)} for value in cur_stats.fetchall()]

        if result:
            df_result = pd.DataFrame.from_dict(result)
        else:
            df_result = pd.DataFrame(columns=['none'])
        conn.close()
        return df_result

    except mariadb.Error as e:
        st.error(f"Error connecting to MariaDB Platform: {e}")
        return None
        #sys.exit(1)


@st.cache_data
def get_dataset_for_pearson(datasetname, heatmap_relation):
    if datasetname == 'CARER':
        df = get_table_for_pearson(datasetname, heatmap_relation, 'heatmap_carer')
    elif datasetname == 'GoEmotions':
        df = get_table_for_pearson(datasetname, heatmap_relation, 'heatmap_goemotions')
    elif datasetname == 'ISEAR':
        df = get_table_for_pearson(datasetname, heatmap_relation, 'heatmap_isear')
    elif datasetname == 'TEC':
        df = get_table_for_pearson(datasetname, heatmap_relation, 'heatmap_tec')
    elif datasetname == 'SemEval2019 Task3':
        df = get_table_for_pearson(datasetname, heatmap_relation, 'heatmap_se')
    else:
        df = pd.DataFrame()
    return df


@st.cache_data
def get_df_all_source_counts():
    df_source = pd.read_csv(dataset_dir+'/data_source.csv', encoding='UTF-8', sep='|')
    df_source = df_source[['source','counts']]
    return df_source


@st.cache_data
def get_dataset_jd_dc(dataset_option, heatmap_relation):
    try:
        conn = mariadb.connect(
            user=mariadb_user,
            password=mariadb_pass,
            host="localhost",
            port=3306,
            database="vis_emotif"
        )

        # Get Cursor
        cur_stats = conn.cursor()

        sql_code = "SELECT * FROM data_jd where source = ?"

        cur_stats.execute(sql_code, ([dataset_option]))

        columns = cur_stats.description 
        result = [{columns[index][0]:column for index, column in enumerate(value)} for value in cur_stats.fetchall()]

        if result:
            df_result = pd.DataFrame.from_dict(result)
            df_result.reset_index(inplace=True, drop=True)
        else:
            df_result = pd.DataFrame(columns=['none'])
        conn.close()
        return df_result

    except mariadb.Error as e:
        st.error(f"Error connecting to MariaDB Platform: {e}")
        return None

@st.cache_data
def get_texts(datasetname, feature, min_textsize, max_textsize):
    try:
        conn = mariadb.connect(
            user=mariadb_user,
            password=mariadb_pass,
            host="localhost",
            port=3306,
            database="vis_emotif"
        )

        if feature == '#Lines':
            columns_str = "`text`,`emotion`,`source`,`length`"
        elif feature == '#Duplicated Texts':
            columns_str = "`text`,`emotion`,`source`,`length`"
        elif feature == 'Average #Sentences/Line':
            columns_str = "`text`,`emotion`,`source`,`length`,`n_sents`"
        elif feature == 'Average #Words/Line':
            columns_str = "`text`,`emotion`,`source`,`length`,`n_words`"
        elif feature == 'Average TTR/Line':
            columns_str = "`text`,`emotion`,`source`,`length`,`diversity_ttr`"
        elif feature == 'Avg. Flesch Reading Ease/Line':
            columns_str = "`text`,`emotion`,`source`,`length`,`flesch_reading_ease`"
        elif feature == '#Entities':
            columns_str = "`text`,`emotion`,`source`,`length`,`n_ents`"

        # Get Cursor
        cur_texts = conn.cursor()        

        if datasetname == 'All Datasets':

            if feature == '#Duplicated Texts':
                sql_text = "SELECT "+columns_str+" FROM data_text_emotion_source_dup WHERE `length` >= "+str(min_textsize)+" AND `length` < "+str(max_textsize)+" LIMIT 100"
            else:    
                sql_text = "(SELECT "+columns_str+" FROM data_text_emotion_source_stats WHERE `source`='CARER' AND `length` >= "+str(min_textsize)+" AND `length` < "+str(max_textsize)+" LIMIT 20) " \
                "UNION ALL " \
                "(SELECT "+columns_str+" FROM data_text_emotion_source_stats WHERE `source`='GoEmotions' AND `length` >= "+str(min_textsize)+" AND `length` < "+str(max_textsize)+" LIMIT 20) " \
                "UNION ALL " \
                "(SELECT "+columns_str+" FROM data_text_emotion_source_stats WHERE `source`='ISEAR' AND `length` >= "+str(min_textsize)+" AND `length` < "+str(max_textsize)+" LIMIT 20) " \
                "UNION ALL " \
                "(SELECT "+columns_str+" FROM data_text_emotion_source_stats WHERE `source`='SemEval2019 Task3' AND `length` >= "+str(min_textsize)+" AND `length` < "+str(max_textsize)+" LIMIT 20) " \
                "UNION ALL " \
                "(SELECT "+columns_str+" FROM data_text_emotion_source_stats WHERE `source`='TEC' AND `length` >= "+str(min_textsize)+" AND `length` < "+str(max_textsize)+" LIMIT 20) "

            cur_texts.execute(sql_text)
        else:
            if feature == '#Duplicated Texts':
                sql_text = "SELECT "+columns_str+" FROM data_text_emotion_source_dup WHERE `source`=? AND `length` >= "+str(min_textsize)+" AND `length` < "+str(max_textsize)+" LIMIT 100"
            else:
                sql_text = "SELECT "+columns_str+" FROM data_text_emotion_source_stats WHERE `source`=? AND `length` >= "+str(min_textsize)+" AND `length` < "+str(max_textsize)+" LIMIT 100"

            cur_texts.execute(sql_text, [datasetname] )

        columns = cur_texts.description 
        result = [{columns[index][0]:column for index, column in enumerate(value)} for value in cur_texts.fetchall()]

        df_result = pd.DataFrame.from_dict(result)
        conn.close()
        return df_result

    except mariadb.Error as e:
        st.error(f"Error connecting to MariaDB Platform: {e}")
        return None
        #sys.exit(1)


@st.cache_data
def get_entities_data(dataset_option):
    if dataset_option == 'All Datasets':
        df = pd.read_csv(dataset_dir+'/ents_source_counts10.csv', encoding='UTF-8', sep='|') 
    else:
        df = pd.read_csv(dataset_dir+'/ents_source_emotion_counts3.csv', encoding='UTF-8', sep='|') 
    return df
