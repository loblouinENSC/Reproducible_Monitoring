import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output
import numpy as np

# --- Configuration ---
TOILET_LOG_FILE = 'rule-toilet.csv'
TOILET_FAILURE_DAYS_FILE = 'sensors_failure_days/toilet_failure_days.csv'
APP_TITLE = "Toilet Activity Viewer"
TEXT_COLOR = 'white'
BACKGROUND_COLOR = '#111111'
DATA1_COLOR = '#36A0EB' #blue
DATA2_COLOR = '#43D37B' # green
DATA3_COLOR = '#EB9636' # Orange
FAILURE_MARKER_COLOR = '#F14864' # Rouge
DATAMONTH_COLOR = DATA1_COLOR
DATAMONTH2_COLOR = DATA2_COLOR


# --- Graph Configuration ---
LEGEND = dict(orientation="h",yanchor="bottom",y=1.1,xanchor="center",x=0.5)
MARGIN_CHART = dict(l=70, r=70, b=70, t=150, pad=3)

TITLE_X = 0.06
TITLE_Y = 0.92


# --- Data Loading and Processing Function --- 
def get_toilet_data():
    try:
        activity = pd.read_csv(TOILET_LOG_FILE, delimiter=';', decimal=",",
                                names=["date", "annotation", "activity_count", "duration"],
                                parse_dates=["date"], index_col="date")
        activity['duration_min'] = activity['duration'] / 60
    except FileNotFoundError:
        print(f"Error: '{TOILET_LOG_FILE}' not found.")
        activity = pd.DataFrame(columns=["annotation", "activity_count", "duration","duration_min"],
                                 index=pd.to_datetime([]))
        activity.index.name = 'date'
    except Exception as e:
        print(f"Error loading {TOILET_LOG_FILE}: {e}")
        activity = pd.DataFrame(columns=["annotation", "activity_count", "duration","duration_min"],
                                 index=pd.to_datetime([]))
        activity.index.name = 'date'

    if not activity.empty:
        activity_daily_intermediate = activity.resample('D').agg(
            activity_count_sum_daily=('activity_count', 'sum'),
            duration_min_sum_daily=('duration_min', 'sum')
        )
    else:
        activity_daily_intermediate = pd.DataFrame(
            columns=['activity_count_sum_daily', 'duration_min_sum_daily'],
            index=pd.to_datetime([])
        )
        activity_daily_intermediate.index.name = 'date'

    if not activity.empty:
        activity_daily_for_graph = activity.resample('D').agg(
            activity_count_sum=('activity_count', 'sum'),
            duration_min_sum=('duration_min', 'sum')
        ).reset_index()
        activity_daily_for_graph['year_month'] = activity_daily_for_graph['date'].dt.to_period('M').astype(str)
    else:
        activity_daily_for_graph = pd.DataFrame(columns=['date', 'activity_count_sum', 'duration_min_sum', 'year_month'])
        activity_daily_for_graph = activity_daily_for_graph.astype({'date': 'datetime64[ns]'})


    if not activity_daily_intermediate.empty:
        monthly_agg_input = activity_daily_intermediate.copy()
        monthly_agg_input['duration_min_sum_daily_for_avg'] = monthly_agg_input['duration_min_sum_daily'].replace(0, np.nan)
        activity_monthly = monthly_agg_input.resample('ME').agg(
            duration_min_mean_of_daily_totals=('duration_min_sum_daily_for_avg', 'mean'),
            duration_min_sem_of_daily_totals=('duration_min_sum_daily_for_avg', 'sem'),
            activity_count_sum_monthly=('activity_count_sum_daily', 'sum')
        ).reset_index()
        activity_monthly = activity_monthly.rename(columns={
            'duration_min_mean_of_daily_totals': 'duration_min_mean',
            'duration_min_sem_of_daily_totals': 'duration_min_sem',
            'activity_count_sum_monthly': 'activity_count_sum'
        })
        if pd.api.types.is_datetime64_any_dtype(activity_monthly['date']):
            activity_monthly['days_in_month'] = activity_monthly['date'].dt.daysinmonth
            activity_monthly['activity_count_mean_daily'] = np.where(
                activity_monthly['days_in_month'] > 0,
                activity_monthly['activity_count_sum'] / activity_monthly['days_in_month'],0)
        else:
            activity_monthly['days_in_month'] = 0
            activity_monthly['activity_count_mean_daily'] = 0
    else:
        activity_monthly = pd.DataFrame(columns=['date', 'activity_count_sum', 'duration_min_mean',
                                                 'duration_min_sem', 'days_in_month',
                                                 'activity_count_mean_daily'],
                                           index=pd.to_datetime([]))
        if not activity_monthly.empty: # Should be empty here, but for safety
             activity_monthly = activity_monthly.reset_index().rename(columns={'index': 'date'})
        else: # Ensure 'date' column exists if df is truly empty and reset_index not applicable
            if 'date' not in activity_monthly.columns:
                 activity_monthly['date'] = pd.Series(dtype='datetime64[ns]')


    toilet_failure_daily_markers = pd.DataFrame(columns=['date'])
    try:
        failure_dates_df = pd.read_csv(
            TOILET_FAILURE_DAYS_FILE,
            header=None, 
            names=['date'],
            parse_dates=[0], 
            comment='#'
        )
        toilet_failure_daily_markers = failure_dates_df[['date']].dropna(subset=['date']).copy()
        toilet_failure_daily_markers['date'] = pd.to_datetime(toilet_failure_daily_markers['date'], errors='coerce')
        toilet_failure_daily_markers.dropna(subset=['date'], inplace=True)
    except FileNotFoundError:
        print(f"Avertissement : Fichier des jours d'échec des toilettes '{TOILET_FAILURE_DAYS_FILE}' non trouvé.")
    except pd.errors.EmptyDataError:
        print(f"Avertissement : Fichier des jours d'échec des toilettes '{TOILET_FAILURE_DAYS_FILE}' est vide.")
    except Exception as e:
        print(f"Erreur lors du chargement de '{TOILET_FAILURE_DAYS_FILE}': {e}")

    if not toilet_failure_daily_markers.empty and 'date' in toilet_failure_daily_markers.columns:
        if not toilet_failure_daily_markers.empty:
            toilet_failure_daily_markers['year_month'] = toilet_failure_daily_markers['date'].dt.to_period('M').astype(str)
        else:
            if 'year_month' not in toilet_failure_daily_markers.columns:
                 toilet_failure_daily_markers['year_month'] = pd.Series(dtype='str')
    else:
        if 'year_month' not in toilet_failure_daily_markers.columns:
             toilet_failure_daily_markers['year_month'] = pd.Series(dtype='str')
    
    toilet_failure_monthly_sum_agg = pd.DataFrame(columns=['toilet_failure_days_sum'], index=pd.to_datetime([]))
    toilet_failure_monthly_sum_agg.index.name = 'date'
    if not toilet_failure_daily_markers.empty:
        temp_failure_monthly = toilet_failure_daily_markers.copy()
        temp_failure_monthly['date'] = pd.to_datetime(temp_failure_monthly['date'])
        temp_failure_monthly = temp_failure_monthly.set_index('date')
        temp_failure_monthly['failure_day_count'] = 1
        
        toilet_failure_monthly_sum_agg = temp_failure_monthly.resample('ME').agg(
            toilet_failure_days_sum=('failure_day_count', 'sum')
        )

    if not activity_monthly.empty and 'date' in activity_monthly.columns:
        activity_monthly['date'] = pd.to_datetime(activity_monthly['date']) # Ensure date column is datetime for merge
        activity_monthly = pd.merge(
            activity_monthly,
            toilet_failure_monthly_sum_agg.reset_index(),
            on='date',
            how='outer'
        )
        activity_monthly['toilet_failure_days_sum'] = activity_monthly['toilet_failure_days_sum'].fillna(0).astype(int)
    elif not toilet_failure_monthly_sum_agg.empty : # activity_monthly était vide, mais on a des données d'échec
        activity_monthly = toilet_failure_monthly_sum_agg.reset_index()
        activity_monthly['toilet_failure_days_sum'] = activity_monthly['toilet_failure_days_sum'].fillna(0).astype(int)
        # S'assurer que les autres colonnes attendues existent
        for col in ['activity_count_sum', 'duration_min_mean', 'duration_min_sem', 'days_in_month', 'activity_count_mean_daily']:
            if col not in activity_monthly.columns: activity_monthly[col] = np.nan if 'mean' in col or 'sem' in col else 0
    else: # Les deux sont vides ou activity_monthly n'a pas de colonne 'date'
        if 'toilet_failure_days_sum' not in activity_monthly.columns:
             activity_monthly['toilet_failure_days_sum'] = 0


    if 'date' in activity_monthly.columns and not activity_monthly.empty:
        valid_dates_monthly = activity_monthly['date'].notna()
        activity_monthly['month_label'] = '' 
        if valid_dates_monthly.any():
            activity_monthly.loc[valid_dates_monthly, 'month_label'] = activity_monthly.loc[valid_dates_monthly, 'date'].dt.strftime('%m/%y')
    else:
        if 'month_label' not in activity_monthly.columns:
             activity_monthly['month_label'] = pd.Series(dtype='str')
        if 'toilet_failure_days_sum' not in activity_monthly.columns:
             activity_monthly['toilet_failure_days_sum'] = 0
        if 'date' not in activity_monthly.columns: 
            activity_monthly['date'] = pd.Series(dtype='datetime64[ns]')


    return activity_daily_for_graph, activity_monthly, toilet_failure_daily_markers


# --- Figure Creation Function ---
def create_toilet_figure(daily_data, monthly_data, daily_failure_markers, scale, selected_month):
    fig = go.Figure()
    fig.update_layout(
        template='plotly_dark', paper_bgcolor=BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR,
        font=dict(color=TEXT_COLOR), title=dict(font=dict(color=TEXT_COLOR)),
        legend=dict(font=dict(color=TEXT_COLOR)),
        xaxis=dict(title=dict(font=dict(color=TEXT_COLOR)), tickfont=dict(color=TEXT_COLOR), gridcolor='rgba(255, 255, 255, 0.1)'),
        yaxis=dict(title=dict(font=dict(color=TEXT_COLOR)), tickfont=dict(color=TEXT_COLOR), gridcolor='rgba(255, 255, 255, 0.1)'),
        yaxis2=dict(title=dict(font=dict(color=TEXT_COLOR)), tickfont=dict(color=TEXT_COLOR), gridcolor='rgba(255, 255, 255, 0.1)'),
        margin = MARGIN_CHART,
        hoverlabel=dict(font_size=16, font_color="white", namelength=-1)
    )

    if scale == 'year':
        df_monthly = monthly_data
        if not df_monthly.empty:
            fig.add_trace(go.Bar(x=df_monthly['month_label'], y=df_monthly['duration_min_mean'], name="Durée moyenne (min)", error_y=dict(type='data', array=df_monthly['duration_min_sem']), marker_color=DATA1_COLOR))
            fig.add_trace(go.Scatter(x=df_monthly['month_label'], y=df_monthly['activity_count_sum'], name="Passages totaux", yaxis='y2', mode='lines+markers', line=dict(color=DATA2_COLOR)))
            if 'activity_count_mean_daily' in df_monthly.columns:
                fig.add_trace(go.Scatter(x=df_monthly['month_label'], y=df_monthly['activity_count_mean_daily'], name="Passages moyens/jour", yaxis='y2', mode='lines+markers', line=dict(color=DATA3_COLOR)))
            
            if 'toilet_failure_days_sum' in df_monthly.columns:
                fig.add_trace(go.Scatter(
                    x=df_monthly['month_label'], 
                    y=df_monthly['toilet_failure_days_sum'], 
                    name="Jours échec capteur", 
                    yaxis='y2', 
                    mode='lines+markers', 
                    line=dict(color=FAILURE_MARKER_COLOR, dash='dot')
                ))
            
            y2_max_val = 5 
            y2_columns_to_check = ['activity_count_sum', 'activity_count_mean_daily', 'toilet_failure_days_sum']
            current_max = 0
            for col in y2_columns_to_check:
                if col in df_monthly.columns and pd.notna(df_monthly[col].max()):
                    current_max = max(current_max, df_monthly[col].max())
            if current_max > 0:
                y2_max_val = current_max * 1.1


            fig.update_layout(
                title=dict(text="Vue Annuelle : Activité mensuelle", x = TITLE_X, y = TITLE_Y),
                xaxis=dict(title=dict(text="Mois")),
                yaxis=dict(title=dict(text="Durée moyenne (min)"), tickfont=dict(color=DATA1_COLOR)),
                yaxis2=dict(title=dict(text="Comptes"), tickfont=dict(color=DATA2_COLOR), overlaying='y', side='right', range=[0, y2_max_val]),
                legend=LEGEND
            )
        else:
            fig.update_layout(title=dict(text="Yearly View: No monthly data available"))

    elif scale == 'month' and selected_month:
        df_daily_activity = daily_data[daily_data['year_month'] == selected_month] if not daily_data.empty else pd.DataFrame()

        df_daily_failure_filtered = pd.DataFrame()
        if not daily_failure_markers.empty and 'year_month' in daily_failure_markers.columns:
            filtered_failures = daily_failure_markers[daily_failure_markers['year_month'] == selected_month]
            if not filtered_failures.empty:
                df_daily_failure_filtered = filtered_failures
        
        all_relevant_dates = []
        if not df_daily_activity.empty:
            all_relevant_dates.extend(df_daily_activity['date'].tolist())
        if not df_daily_failure_filtered.empty:
            all_relevant_dates.extend(df_daily_failure_filtered['date'].tolist())
        
        unique_display_dates = []
        if all_relevant_dates:
            unique_display_dates = sorted(list(set(all_relevant_dates)))

        if not df_daily_activity.empty:
            if 'duration_min_sum' in df_daily_activity.columns:
                fig.add_trace(go.Bar(x=df_daily_activity['date'], y=df_daily_activity['duration_min_sum'], name="Durée totale (min)", yaxis='y1', marker_color = DATAMONTH_COLOR))
            fig.add_trace(go.Scatter(x=df_daily_activity['date'], y=df_daily_activity['activity_count_sum'],name="Passages par jour",yaxis='y2', mode='lines+markers', line=dict(color=DATAMONTH2_COLOR) ))

        if not df_daily_failure_filtered.empty:
            fig.add_trace(go.Scatter(
                x=df_daily_failure_filtered['date'], 
                y=[1.0] * len(df_daily_failure_filtered), 
                name="Échec capteur", 
                mode='markers', 
                marker=dict(color=FAILURE_MARKER_COLOR, size=10, symbol='x'),
                
                yaxis='y1' 
            ))
        
        if not df_daily_activity.empty or not df_daily_failure_filtered.empty:
            fig.update_layout(
                title=dict(text=f"Vue Journalière : {selected_month}", x=TITLE_X, y=TITLE_Y),
                xaxis=dict(title=dict(text="Jour"), 
                           tickformat='%d',
                           tickmode='array', 
                           tickvals=unique_display_dates, 
                           ticktext=[d.strftime('%d') for d in unique_display_dates] if unique_display_dates else [],
                           dtick="D1" if not unique_display_dates else None
                          ),
                yaxis=dict(title=dict(text="Durée totale (min)"),range=[0, 40],tickfont=dict(color=DATAMONTH_COLOR)),
                yaxis2=dict(title=dict(text="Nombre de passages"),range=[0, 30], tickfont=dict(color=DATAMONTH2_COLOR),overlaying='y', side='right',showgrid=False),
                legend=LEGEND
            )
        else: 
             fig.update_layout(title=dict(text=f"Monthly View: No data or sensor failures for {selected_month}", x=TITLE_X, y=TITLE_Y))

    elif scale == 'month' and not selected_month:
        fig.update_layout(title=dict(text="Monthly View: Please select a month"))

    if not fig.data:
        fig.update_layout(title=dict(text="Aucune donnée à afficher"))

    return fig

# --- Standalone App Execution ---
if __name__ == '__main__':
    activity_daily_data, activity_monthly_data, toilet_failure_markers_data = get_toilet_data() 

    available_months = []
    temp_months_set = set()
    if not activity_daily_data.empty and 'year_month' in activity_daily_data.columns:
        temp_months_set.update(activity_daily_data['year_month'].unique())
    if not toilet_failure_markers_data.empty and 'year_month' in toilet_failure_markers_data.columns:
        valid_year_months = toilet_failure_markers_data['year_month'].dropna().unique()
        temp_months_set.update(valid_year_months)
    
    if temp_months_set:
        available_months = sorted(list(temp_months_set))


    app = Dash(__name__)
    app.title = APP_TITLE
    app.layout = html.Div(style={'backgroundColor': BACKGROUND_COLOR, 'color': TEXT_COLOR, 'padding': '20px'}, children=[
        html.H2(APP_TITLE, style={'textAlign': 'center', 'marginBottom': '30px'}),
        html.Div([
            html.Label("Select view scale:", style={'marginRight': '10px'}),
            dcc.RadioItems(
                id='scale-selector',
                options=[{'label': 'Year View (Monthly)', 'value': 'year'}, {'label': 'Month View (Daily)', 'value': 'month'}],
                value='year',
                labelStyle={'display': 'inline-block', 'marginRight': '20px'},
                inputStyle={'marginRight': '5px'}
            ),
        ], style={'marginBottom': '20px', 'textAlign': 'center'}),
        html.Div(id='month-dropdown-container', children=[
            html.Label("Select Month:", style={'marginRight': '10px'}),
            dcc.Dropdown(
                id='month-dropdown',
                options=[{'label': m, 'value': m} for m in available_months],
                value=available_months[0] if len(available_months) > 0 else None,
                clearable=False,
                style={'width': '200px', 'display': 'inline-block', 'color': '#333'}
            )
        ], style={'display': 'none', 'textAlign': 'center', 'marginBottom': '20px'}),
        dcc.Graph(id='activity-graph', style={'height': '60vh'})
    ])

    @app.callback(
        Output('month-dropdown-container', 'style'),
        Input('scale-selector', 'value')
    )
    def toggle_month_dropdown(scale):
        display = 'block' if scale == 'month' else 'none'
        return {'display': display, 'textAlign': 'center', 'marginBottom': '20px'}

    @app.callback(
        Output('activity-graph', 'figure'),
        Input('scale-selector', 'value'),
        Input('month-dropdown', 'value')
    )
    def update_graph_standalone(scale, selected_month):
        return create_toilet_figure(activity_daily_data, activity_monthly_data, toilet_failure_markers_data, scale, selected_month) 

    app.run(debug=True)