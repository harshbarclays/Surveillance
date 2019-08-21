"""
FX Twitter Project using Streaming Data
Created by Aditya Gandhi
"""


import pandas as pd
from bokeh.models.widgets import DataTable, TableColumn, Button, Div, HTMLTemplateFormatter
from bokeh.models import Span, HoverTool, ColumnDataSource, Select, Range1d, LinearAxis, DateFormatter, NumberFormatter,\
    Label, StringEditor, Legend, FuncTickFormatter, MultiSelect, TextInput
from bokeh.plotting import figure
from bokeh.models.glyphs import ImageURL
import datetime as dt
from bokeh.models.widgets import Tabs, Panel
from bokeh.io import curdoc, export_png
from bokeh.layouts import row, column, layout, Spacer
from bokeh.models.annotations import LegendItem
from bokeh.models.callbacks import CustomJS
import import_tweets_v3 as import_tweets
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from spinner import create_spinner, show_spinner, hide_spinner
from datetime import timedelta
import clean_tweets

stopwords = set(STOPWORDS)
div_spinner = create_spinner()

numFmt = NumberFormatter(format="0.0000")
largenumFmt = NumberFormatter(format="0,000")
dateFmt = DateFormatter(format="ddMyy")
percFmt = NumberFormatter(format='0.000 %')
xaxisRange = Range1d(start=None, end=None)

# Bokeh Time Function
def bokeh_time(dtstr):
    if dtstr != None:
        return dtstr.value / 1e6
    else:
        return 0

# Filter the data based on datetime
def get_df_filtered(df, datetime_val):
    datetime_next = datetime_val + dt.timedelta(days=1)
    print(datetime_next)
    filtered_df = df.loc[(df['DateTime'] >= datetime_val) & (df['DateTime'] < datetime_next)].copy()
    return filtered_df


def create_wordcloud(data, title=None):
    wordcloud = WordCloud(
        background_color='lightblue',
        stopwords=stopwords,
        max_words=50,
        max_font_size=40,
        scale=3,
        random_state=1  # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(8, 8))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()
    wordcloud.to_file("Codes/static/image_2.png")


def update_plot():
    show_spinner(div_spinner)
    ccy_pair = ccy_pair_menu.value
    curr_date_up = from_date_menu.value
    curr_time_up = '00:00:00'
    curr_datetime_string_up = str(curr_date_up) + " " + str(curr_time_up)
    curr_datetime_up = pd.to_datetime(curr_datetime_string_up, format='%Y.%m.%d %H:%M:%S')

    data_up = pd.read_excel('Codes/' + str(ccy_pair) + '.xlsx')
    data_up['DateTime'] = pd.to_datetime(data_up['DateTime'])
    data_up['HLC_avg'] = (data_up['High'] + data_up['Low'] + data_up['Close']) / 3
    plot_data_up = get_df_filtered(data_up, curr_datetime_up)

    minRate_up = 0.999 * plot_data_up['Close'].min()
    maxRate_up = 1.001 * plot_data_up['Close'].max()


    plot.title.text = ccy_pair + ' Currency Exchange Rate'
    plot.y_range.start = minRate_up
    plot.y_range.end = maxRate_up
    plot_data_source_up = ColumnDataSource(plot_data_up)
    plot_data_source.data.update(plot_data_source_up.data)

    tweets_df_up = import_tweets.get_tweets_currency(ccy_pair + ' OR ' + ccy_pair[0:3] + '/' + ccy_pair[3:])
    tweets_df_up.to_csv('tweets_info.csv')
    create_wordcloud(tweets_df_up['Message'])

    most_repeated_up, hashtags_up, lda_results_up, log_like_up, perp_up = \
        clean_tweets.clean_all_tweets_apply_model(tweets_df_up, ccy_pair[0:3].lower(),ccy_pair[3:].lower(), ccy_pair.lower())

    data_table_source_up = ColumnDataSource(tweets_df_up)
    data_table_source.data.update(data_table_source_up.data)

    data_table_lda_source_up = ColumnDataSource(lda_results_up)
    repeated_source_up = ColumnDataSource(most_repeated_up)
    df_perp_up = pd.DataFrame([[log_like_up, perp_up]], columns=['log_likelihood', 'perplexity'])
    lda_perf_source_up = ColumnDataSource(df_perp_up)

    data_table_lda_source.data.update(data_table_lda_source_up.data)
    repeated_source.data.update(repeated_source_up.data)
    lda_perf_source.data.update(lda_perf_source_up.data)

    hide_spinner(div_spinner)

def update_secondary_axis():
    current_date_ax = from_date_menu.value
    date_start = pd.to_datetime(current_date_ax, format='%Y.%m.%d')
    date_end = date_start + timedelta(hours=24)
    plot.x_range.start = bokeh_time(date_start)
    plot.x_range.end = bokeh_time(date_end)


# def update_wordcloud(a):
#     print("Going inside wordcloud function")
#     a.visible = False
#     create_wordcloud()





# Creating the Dropdowns / Filter
ccy_pair_menu = Select(options=['EURUSD', 'GBPUSD', 'USDJPY', 'EURGBP', 'EURJPY'], value='EURUSD', title='Currency Pair (Base/Quote)')
from_date_menu = TextInput(value=str('2019.05.17'), title="View Date (mm.dd.yyyy):")
time_menu = TextInput(value=str('17:00:00'), title="Time Snapshot (24-hr format):")
button_go = Button(label="Refresh", width=100, height=30)
button_set_axis = Button(label="Update Axis", width=100, height=30)
button_rec_cloud = Button(label="Refresh WordCloud", width=150, height=30)
curr_date = from_date_menu.value
curr_time = '00:00:00'
TOOLS ='pan,wheel_zoom,box_zoom,tap,save,lasso_select,xbox_select,reset'


# Convert to datetime
curr_datetime_string = str(curr_date) + " " + str(curr_time)
curr_datetime = pd.to_datetime(curr_datetime_string, format='%Y.%m.%d %H:%M:%S')
print(curr_datetime)

# Read in data for the latest month from the excel files
data = pd.read_excel('Codes/EURUSD.xlsx')
data['DateTime'] = pd.to_datetime(data['DateTime'])
data['HLC_avg'] = (data['High'] + data['Low'] + data['Close']) / 3
print(data.dtypes)
print(data)
plot_data = get_df_filtered(data, curr_datetime)

# Use the right metrics for creating the graph
minRate = 0.999*plot_data['Close'].min()
maxRate = 1.001*plot_data['Close'].max()


# Set up the overall plot
plot = figure(
    plot_width=800,
    plot_height=450,
    toolbar_location = 'below',
    title = 'EURUSD Currency Exchange Rate',
    x_axis_label = 'DateTime',
    y_axis_label = 'Exchange Rate',
    x_axis_type = 'datetime',
    tools = TOOLS,
    active_drag = 'xbox_select',
    y_range = Range1d(start=minRate, end=maxRate)
)

# Create the ColumnDataSource for this plot
plot_data_source = ColumnDataSource(plot_data)

# Create the exchange rate line for the close and the HLC average line
plot.line(x='DateTime', y='Close', color='navy', legend='Close Pos (End-min)', source = plot_data_source)
#plot.circle(x='DateTime', y='HLC_avg', color='darkgreen', legend='Avg Pos (Intra-min)', source = plot_data_source)

# Adjust a few overall options
plot.title.align = 'center'
plot.toolbar_location = 'right'

# new_legend = plot.legend[0]
# new_legend.click_policy = 'hide'
# plot.legend[0].plot=None
# plot.add_layout(new_legend, 'right')


# Get real-time Twitter Data
tweets_df = import_tweets.get_tweets_currency('EURUSD OR EUR/USD')
tweets_df.to_csv('tweets_info.csv')
create_wordcloud(tweets_df['Message'])

# Clean Twitter Data
most_repeated, hashtags, lda_results, log_like, perp = clean_tweets.clean_all_tweets_apply_model(tweets_df, 'eur', 'usd', 'eurusd')



# Create a DataTable for the Twitter Data to be displayed
tweet_data_columns = [
    TableColumn(field='Source', title='Username'),
    TableColumn(field='Created At', title='Tweet DateTime', formatter=DateFormatter(format='%Y-%m-%d %H:%M:%S'), editor=StringEditor(), width=350),
    TableColumn(field='Message', title='Message', width=2000),
    TableColumn(field='Retweet Count', title = 'Retweet Count'),
    TableColumn(field='Favorite Count', title = 'Favorite Count'),
    TableColumn(field='is_retweet', title='Re-Tweet'),
    TableColumn(field='mentioned', title='Mentions'),
    TableColumn(field='hashtags', title='Hashtags'),
]

data_table_source = ColumnDataSource(tweets_df)
tweets_table = DataTable(source=data_table_source, columns=tweet_data_columns, width=1200, height=400)


# Create a DataTable for the LDA Results
lda_data_columns = [
    TableColumn(field='Topic 1 words', title='Topic 1 Words'),
    TableColumn(field='Topic 1 weights', title='Topic 1 Weights'),
    TableColumn(field='Topic 2 words', title='Topic 2 Words'),
    TableColumn(field='Topic 2 weights', title='Topic 2 Weights'),
    TableColumn(field='Topic 3 words', title='Topic 3 Words'),
    TableColumn(field='Topic 3 weights', title='Topic 3 Weights'),
    TableColumn(field='Topic 4 words', title='Topic 4 Words'),
    TableColumn(field='Topic 4 weights', title='Topic 4 Weights'),
    TableColumn(field='Topic 5 words', title='Topic 5 Words'),
    TableColumn(field='Topic 5 weights', title='Topic 5 Weights'),
]
data_table_lda_source = ColumnDataSource(lda_results)
lda_table = DataTable(source=data_table_lda_source, columns=lda_data_columns, width=700, height=400)

# Create a DataTable for the LDA Performance Stats
lda_perf_columns = [
    TableColumn(field='log_likelihood', title='Log-Likelihood'),
    TableColumn(field='perplexity', title='Perplexity'),
]
df_perp = pd.DataFrame([[log_like, perp]], columns=['log_likelihood', 'perplexity'])
lda_perf_source = ColumnDataSource(df_perp)
lda_perf_table = DataTable(source=lda_perf_source, columns=lda_perf_columns, width = 400)

# Create a DataTable for the Most Repeated Tweets
repeated_columns = [
    TableColumn(field='Message', title='Message'),
    TableColumn(field='Freq', title='Frequency'),
]
repeated_source = ColumnDataSource(most_repeated)
repeated_table = DataTable(source=repeated_source, columns=repeated_columns, width = 400)

# Plotting an image
#image_cloud = "Codes/static/image.png"
#url = ColumnDataSource(dict(url=[image_cloud]))

x_range = (-20,-5) # could be anything - e.g.(0,1)
y_range = (20,25)
p = figure(x_range=x_range, y_range=y_range, plot_width=450,plot_height=350)
img_path = 'Codes/static/image_2.png'
#source = ColumnDataSource(dict(url=[img_path]))
a = p.image_url(url=[img_path],x=x_range[0],y=y_range[1],w=x_range[1]-x_range[0],h=y_range[1]-y_range[0])
#dummy = p.circle([1],[1], name='dummy', visible=False)


#p.add_glyph(image2)

p.toolbar.logo = None
p.toolbar_location = None
p.xaxis.visible = None
p.yaxis.visible = None
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None
p.outline_line_alpha = 0

# Trying out the Tab / Panel format for Bokeh
tab_0 = Panel(child=tweets_table, title="Twitter Data")
tab_1 = Panel(child=repeated_table, title="Repeated Tweets")
tab_2 = Panel(child=lda_table, title="LDA Analysis")
tab_3 = Panel(child=lda_perf_table, title="Perf. Metrics")

tab_list_0 = Tabs(tabs=[tab_0], sizing_mode='scale_width')
tab_list_1 = Tabs(tabs=[tab_1, tab_2, tab_3], sizing_mode='scale_width')


# Creating a new layout with the title/graph/filters
layout = layout(Div(text="<h2>Forex Market Summary using Twitter and Currency Exchange Rates"),
                row([button_go, button_set_axis, button_rec_cloud, div_spinner]),
                row([ccy_pair_menu, from_date_menu]),
                row([plot, tab_list_1]),
                row([tab_list_0]),
                row([p]),sizing_mode='scale_width')

# Display Plot
curdoc().add_root(layout)
button_go.on_click(update_plot)
button_set_axis.on_click(update_secondary_axis)
#button_rec_cloud.on_click(update_wordcloud(a))
#dummy.glyph.on_change('size', update_wordcloud)















