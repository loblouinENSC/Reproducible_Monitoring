import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output
import numpy as np

# --- Import functions from the refactored visualization scripts ---
def process_daily_data(df):
    if df.empty:
        for col in ['timestamp', 'year_month', 'date', 'hour']:
            if col not in df.columns:
                df[col] = None
        return df

    if 'timestamp' not in df.columns:
        if 'year_month' in df.columns and 'day' in df.columns:
            try:
                df['date_str_temp'] = df['year_month'] + '-' + df['day'].astype(str).str.zfill(2)
                df['timestamp'] = pd.to_datetime(df['date_str_temp'], errors='coerce')
                df.drop(columns=['date_str_temp'], inplace=True)
            except Exception:
                df['timestamp'] = pd.NaT
        elif 'date' in df.columns:
             df['timestamp'] = pd.to_datetime(df['date'], errors='coerce')
        else:
            df['timestamp'] = pd.NaT

    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
         df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    if 'timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        if 'year_month' not in df.columns:
            df['year_month'] = df['timestamp'].dt.strftime('%Y-%m')
        if 'date' not in df.columns:
            df['date'] = df['timestamp'].dt.strftime('%Y-%m-%d') # Ensure this is YYYY-MM-DD string
        if 'hour' not in df.columns:
            df['hour'] = df['timestamp'].dt.hour
    else:
        for col, default_val in [('year_month', None), ('date', None), ('hour', None)]:
            if col not in df.columns:
                df[col] = default_val
    return df

try:
    from visualize_toilet_new import get_toilet_data, create_toilet_figure
    print("Successfully imported toilet functions.")
except ImportError as e:
    print(f"Error importing from visualize_toilet_new.py: {e}")
    def get_toilet_data(): return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    def create_toilet_figure(*args): return go.Figure().update_layout(title_text="Error loading toilet module (stub)")

try:
    from visualize_sleep_quiet_new import get_sleep_data, create_sleep_figure
    print("Successfully imported sleep functions.")
except ImportError as e:
    print(f"Error importing from visualize_sleep_quiet_new.py: {e}")
    def get_sleep_data(): return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    def create_sleep_figure(*args): return go.Figure().update_layout(title_text="Error loading sleep module (stub)")

try:
    from visualize_outing_new import get_outings_data, create_outings_figure
    print("Successfully imported outings functions.")
except ImportError as e:
    print(f"Error importing from visualize_outing_new.py: {e}")
    def get_outings_data(): return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    def create_outings_figure(*args): return go.Figure().update_layout(title_text="Error loading outings module (stub)")


# --- Configuration Globale ---
TEXT_COLOR = 'white'
BACKGROUND_COLOR = '#111111'

TOILET_APP_NAME = "Toilet Activity"
SLEEP_APP_NAME = "Sleep Activity"
OUTINGS_APP_NAME = "Outings Activity"

# --- Chargement Initial des Données ---
print("Loading initial data...")

# Initialize global dataframes
g_toilet_raw_ts_df, g_toilet_daily_agg_df, g_toilet_monthly_agg_df, g_toilet_failure_markers_df = pd.DataFrame(), process_daily_data(pd.DataFrame()), pd.DataFrame(), pd.DataFrame()
g_sleep_raw_ts_df, g_sleep_daily_agg_df, g_sleep_monthly_agg_df, g_sleep_bed_failure_daily_markers = pd.DataFrame(), process_daily_data(pd.DataFrame()), pd.DataFrame(), pd.DataFrame()
g_outings_raw_ts_df, g_outings_daily_agg_df, g_outings_monthly_agg_df, g_outings_door_failure_daily_markers = pd.DataFrame(), process_daily_data(pd.DataFrame()), pd.DataFrame(), pd.DataFrame()

try:
    df1, df2, df3, df4 = get_toilet_data()
    g_toilet_raw_ts_df = df1.copy() if not df1.empty else pd.DataFrame()
    g_toilet_daily_agg_df = process_daily_data(df2.copy() if not df2.empty else pd.DataFrame())
    g_toilet_monthly_agg_df = df3.copy() if not df3.empty else pd.DataFrame()
    g_toilet_failure_markers_df = df4.copy() if not df4.empty else pd.DataFrame()
    print(f"Toilet data loaded: Raw shape={g_toilet_raw_ts_df.shape}, Daily agg shape={g_toilet_daily_agg_df.shape}, Monthly agg shape={g_toilet_monthly_agg_df.shape}, Failures shape={g_toilet_failure_markers_df.shape}")
except ValueError as ve:
    print(f"Error unpacking data from get_toilet_data (expected 4 values): {ve}")
except Exception as e:
    print(f"Error running get_toilet_data: {e}")

try:
    df1, df2, df3, df4 = get_sleep_data()
    g_sleep_raw_ts_df = df1.copy() if not df1.empty else pd.DataFrame()
    g_sleep_daily_agg_df = process_daily_data(df2.copy() if not df2.empty else pd.DataFrame())
    g_sleep_monthly_agg_df = df3.copy() if not df3.empty else pd.DataFrame()
    g_sleep_bed_failure_daily_markers = df4.copy() if not df4.empty else pd.DataFrame()
    print(f"Sleep data loaded: Raw shape={g_sleep_raw_ts_df.shape}, Daily agg shape={g_sleep_daily_agg_df.shape}, Monthly agg shape={g_sleep_monthly_agg_df.shape}, Failures shape={g_sleep_bed_failure_daily_markers.shape}")
except ValueError as ve:
    print(f"Error unpacking data from get_sleep_data (expected 4 values): {ve}")
except Exception as e:
    print(f"Error running get_sleep_data: {e}")

try:
    df1, df2, df3, df4 = get_outings_data()
    g_outings_raw_ts_df = df1.copy() if not df1.empty else pd.DataFrame()
    g_outings_daily_agg_df = process_daily_data(df2.copy() if not df2.empty else pd.DataFrame())
    g_outings_monthly_agg_df = df3.copy() if not df3.empty else pd.DataFrame()
    g_outings_door_failure_daily_markers = df4.copy() if not df4.empty else pd.DataFrame()
    print(f"Outings data loaded: Raw shape={g_outings_raw_ts_df.shape}, Daily agg shape={g_outings_daily_agg_df.shape}, Monthly agg shape={g_outings_monthly_agg_df.shape}, Failures shape={g_outings_door_failure_daily_markers.shape}")
except ValueError as ve:
    print(f"Error unpacking data from get_outings_data (expected 4 values): {ve}")
except Exception as e:
    print(f"Error running get_outings_data: {e}")

print("Initial data loading complete.")

# --- Initialisation de l'App Dash ---
app = Dash(__name__)
app.title = "Activity Dashboard Manager"

# --- Définition du Layout de l'App ---
app.layout = html.Div(id="app-container", children=[
    html.H2(app.title, id="app-title"),
    html.Div(id="control-container", children=[
        html.Div(className='selector', children=[
            html.Label("Activity :"),
            dcc.Dropdown(
                id='activity-type-selector',
                options=[
                    {'label': TOILET_APP_NAME, 'value': 'toilet'},
                    {'label': SLEEP_APP_NAME, 'value': 'sleep'},
                    {'label': OUTINGS_APP_NAME, 'value': 'outings'}
                ],
                value='toilet',
                clearable=False,
                className='dash-dropdown'
            ),
        ]),
        html.Div(className='selector', children=[
            html.Label("View scale :"),
            dcc.Dropdown(
                id='scale-selector',
                options=[
                    {'label': 'Year View', 'value': 'year'},
                    {'label': 'Month View', 'value': 'month'},
                    {'label': 'Day View', 'value': 'day'}
                ],
                value='year',
                clearable=False,
                className='dash-dropdown'
            ),
        ]),
        html.Div(id='month-dropdown-container', className='selector', children=[
            html.Label("Select Month:"),
            dcc.Dropdown(
                id='month-dropdown',
                clearable=False,
                className='dash-dropdown'
            )
        ], style={'display': 'none'}),
        html.Div(id='day-dropdown-container', className='selector', children=[
            html.Label("Select Day:"),
            dcc.Dropdown(
                id='day-dropdown',
                clearable=False,
                className='dash-dropdown'
            )
        ], style={'display': 'none'}),
    ]),
    dcc.Graph(id='activity-graph')
])

# --- Callbacks ---
@app.callback(
    Output('month-dropdown-container', 'style'),
    Output('day-dropdown-container', 'style'),
    Input('scale-selector', 'value')
)
def toggle_dropdown_visibility(scale):
    month_style = {'display': 'none'}
    day_style = {'display': 'none'}
    if scale == 'month':
        month_style = {'display': 'flex'}
    elif scale == 'day':
        month_style = {'display': 'flex'}
        day_style = {'display': 'flex'}
    return month_style, day_style

@app.callback(
    Output('month-dropdown', 'options'),
    Output('month-dropdown', 'value'),
    Input('activity-type-selector', 'value')
)
def update_month_dropdown_options(selected_activity_type):
    options = []
    value = None
    df_daily_for_months = pd.DataFrame()

    if selected_activity_type == 'toilet':
        df_daily_for_months = g_toilet_daily_agg_df
    elif selected_activity_type == 'sleep':
        df_daily_for_months = g_sleep_daily_agg_df
    elif selected_activity_type == 'outings':
        df_daily_for_months = g_outings_daily_agg_df

    available_months_set = set()
    if not df_daily_for_months.empty and 'year_month' in df_daily_for_months.columns:
        valid_months = df_daily_for_months['year_month'].dropna()
        if not valid_months.empty:
            available_months_set.update(valid_months.unique())

    if available_months_set:
        available_months_sorted = sorted(list(available_months_set))
        options = [{'label': str(m), 'value': str(m)} for m in available_months_sorted]
        if available_months_sorted:
            value = available_months_sorted[0]
    return options, value

@app.callback(
    Output('day-dropdown', 'options'),
    Output('day-dropdown', 'value'),
    Input('activity-type-selector', 'value'),
    Input('month-dropdown', 'value'),
    Input('scale-selector', 'value')
)
def update_day_dropdown_options(selected_activity_type, selected_month, scale):
    if scale != 'day' or not selected_month:
        return [], None

    options = []
    value = None
    df_daily_for_days = pd.DataFrame()

    if selected_activity_type == 'toilet':
        df_daily_for_days = g_toilet_daily_agg_df
    elif selected_activity_type == 'sleep':
        df_daily_for_days = g_sleep_daily_agg_df
    elif selected_activity_type == 'outings':
        df_daily_for_days = g_outings_daily_agg_df

    if not df_daily_for_days.empty and 'date' in df_daily_for_days.columns and 'year_month' in df_daily_for_days.columns:
        if selected_month and df_daily_for_days['year_month'].notna().any():
            # Ensure 'date' column is string for comparison if selected_month is string
            # Or ensure selected_month is compatible with df_daily_for_days['year_month'] type
            days_in_month_df = df_daily_for_days[df_daily_for_days['year_month'] == selected_month]
            # 'date' in days_in_month_df should be YYYY-MM-DD strings from process_daily_data
            valid_dates = days_in_month_df['date'].dropna()
            if not valid_dates.empty:
                available_days = sorted(valid_dates.unique()) 
                options = []
                for d_str in available_days:
                    try:
                        dt_obj = pd.to_datetime(d_str) # d_str is YYYY-MM-DD
                        label = dt_obj.strftime('%d/%m')
                        options.append({'label': label, 'value': d_str})
                    except ValueError:
                        options.append({'label': d_str, 'value': d_str})
                if available_days:
                    value = available_days[0]
    return options, value

@app.callback(
    Output('activity-graph', 'figure'),
    Input('activity-type-selector', 'value'),
    Input('scale-selector', 'value'),
    Input('month-dropdown', 'value'),
    Input('day-dropdown', 'value')
)
def update_main_graph(activity_type, scale, selected_month, selected_day):
    fig = go.Figure()
    current_month = selected_month if scale in ['month', 'day'] else None
    current_day = selected_day if scale == 'day' else None

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        font=dict(color=TEXT_COLOR)
    )

    try:
        create_fn = None
        df1_pass, df2_pass, df3_pass, df4_pass = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        if activity_type == 'toilet':
            create_fn = create_toilet_figure
            df1_pass = g_toilet_raw_ts_df
            df2_pass = g_toilet_daily_agg_df
            df3_pass = g_toilet_monthly_agg_df
            df4_pass = g_toilet_failure_markers_df
        elif activity_type == 'sleep':
            create_fn = create_sleep_figure
            df1_pass = g_sleep_raw_ts_df
            df2_pass = g_sleep_daily_agg_df
            df3_pass = g_sleep_monthly_agg_df
            df4_pass = g_sleep_bed_failure_daily_markers
        elif activity_type == 'outings':
            create_fn = create_outings_figure
            df1_pass = g_outings_raw_ts_df
            df2_pass = g_outings_daily_agg_df
            df3_pass = g_outings_monthly_agg_df
            df4_pass = g_outings_door_failure_daily_markers
        
        if create_fn:
            fig = create_fn(
                df1_pass,    # Raw/TS data
                df2_pass,    # Daily aggregated data
                df3_pass,    # Monthly aggregated data
                df4_pass,    # Failure markers data
                scale,
                current_month,
                current_day
            )
        else:
            fig.update_layout(title=dict(text="Select an activity type", font=dict(color=TEXT_COLOR)))
            
    except Exception as e:
        print(f"Error creating figure for {activity_type} with scale {scale}: {e}")
        fig.update_layout(
            title=dict(text=f"Error generating graph for {activity_type}: Check console", font=dict(color='red'))
        )

    current_title = fig.layout.title.text if fig.layout and fig.layout.title and fig.layout.title.text else ""
    is_stub_title = "Error loading module (stub)" in current_title

    if not fig.data and (not current_title or is_stub_title):
        title_text = "Veuillez sélectionner une activité et une échelle."
        if scale == 'day':
            if not current_month:
                 title_text = "Veuillez d'abord sélectionner un mois."
            elif not current_day:
                 title_text = "Veuillez sélectionner un jour pour la vue horaire."
        elif scale == 'month' and not current_month:
            title_text = "Veuillez sélectionner un mois pour la vue journalière."
        
        if is_stub_title and not fig.data :
             fig.update_layout(title_text = title_text)
        elif not fig.data :
             fig.update_layout(title_text = title_text)

    fig.update_layout(
        template='plotly_dark', paper_bgcolor=BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR,
        font=dict(color=TEXT_COLOR)
    )
    if "Error generating graph" in current_title:
        fig.update_layout(title=dict(text=current_title, font=dict(color='red')))

    return fig

# --- Exécution de l'App ---
if __name__ == '__main__':
    print("Starting Dash Manager App...")
    app.run(debug=True)

