import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
from datetime import datetime
import re

st.set_page_config(page_title="Direct Deposit Analysis", layout="wide")

# ---------- Upload Zimbabwean Medical Aid List ----------
uploaded_aid_file = st.file_uploader("Upload Medical Aid List (Excel/CSV, optional)", type=["xlsx","csv"])
if uploaded_aid_file:
    if uploaded_aid_file.name.endswith('.xlsx'):
        aid_df = pd.read_excel(uploaded_aid_file)
    else:
        aid_df = pd.read_csv(uploaded_aid_file)
    if 'MedicalAid' in aid_df.columns:
        MEDICAL_AIDS_DYNAMIC = [str(x).upper() for x in aid_df['MedicalAid'].dropna()]
    else:
        MEDICAL_AIDS_DYNAMIC = [str(x).upper() for x in aid_df.iloc[:,0].dropna()]
else:
    MEDICAL_AIDS_DYNAMIC = ["CIMAS", "FIDELITY", "OLD MUTUAL", "FIRST MUTUAL", "MEDISURE", "ZIMRA"]

# ---------- Helper Functions ----------
def clean_amount(x):
    try:
        return float(str(x).replace(",", "").replace("$","").strip())
    except:
        return 0.0

def categorize_payer(payer):
    payer_str = str(payer).upper()
    if any(aid in payer_str for aid in MEDICAL_AIDS_DYNAMIC):
        return "Medical Aid"
    return "Individual"

def aggregate_by_aid(df):
    return (df[df['PAYER_TYPE']=='Medical Aid']
            .groupby('Payer')
            .agg(
                Total_Amount=('Amount','sum'),
                Deposit_Count=('Amount','count'),
                Average_Deposit=('Amount','mean')
            )
            .reset_index()
            .sort_values(by='Total_Amount', ascending=False))

def aggregate_individuals(df):
    return (df[df['PAYER_TYPE']=='Individual']
            .groupby('Payer')
            .agg(
                Total_Amount=('Amount','sum'),
                Deposit_Count=('Amount','count'),
                Average_Deposit=('Amount','mean')
            )
            .reset_index()
            .sort_values(by='Total_Amount', ascending=False))

def generate_comparison(df):
    df['Month'] = df['Date'].dt.to_period('M')
    current_month = df['Month'].max()
    previous_month = current_month - 1

    df_current = df[df['Month'] == current_month]
    df_previous = df[df['Month'] == previous_month]

    current_summary = aggregate_by_aid(df_current)
    previous_summary = aggregate_by_aid(df_previous)

    comparison = current_summary.merge(
        previous_summary,
        on='Payer',
        how='outer',
        suffixes=('_Current', '_Previous')
    ).fillna(0)

    comparison['Difference'] = comparison['Total_Amount_Current'] - comparison['Total_Amount_Previous']
    comparison['Percent_Change'] = comparison.apply(
        lambda x: (x['Difference'] / x['Total_Amount_Previous'] * 100) if x['Total_Amount_Previous']>0 else 0,
        axis=1
    )
    comparison = comparison.sort_values(by='Total_Amount_Current', ascending=False)
    return comparison, current_month, previous_month

def calculate_trends(df, freq='M'):
    df_trend = df.copy()
    df_trend['Period'] = df_trend['Date'].dt.to_period(freq)
    
    medical_trend = (df_trend[df_trend['PAYER_TYPE']=='Medical Aid']
                     .groupby('Period')['Amount'].sum()
                     .reset_index())
    individual_trend = (df_trend[df_trend['PAYER_TYPE']=='Individual']
                        .groupby('Period')['Amount'].sum()
                        .reset_index())
    
    medical_trend['Growth_%'] = medical_trend['Amount'].pct_change().fillna(0)*100
    individual_trend['Growth_%'] = individual_trend['Amount'].pct_change().fillna(0)*100
    
    return medical_trend, individual_trend

# ---------- Streamlit UI ----------
st.title("Zimbabwean Direct Deposit Analysis")

uploaded_file = st.file_uploader("Upload Direct Deposit Excel/CSV", type=["xlsx","csv"])
compare_prev_month = st.checkbox("Compare Previous Month vs Current Month", value=True)

if uploaded_file:
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    # --- Normalize headers ---
    df.columns = [str(c).replace('\xa0',' ').replace('\u200b','').strip() for c in df.columns]

    # Rename common columns
    df = df.rename(columns={
        'Transaction date': 'Date',
        'Transaction description': 'Payer'
    })

    # --- Robust Amount Detection ---
    def find_amount_column(columns):
        candidates = []
        for col in columns:
            norm = re.sub(r'[^A-Z0-9]', '', str(col).upper())
            if 'AMOUNT' in norm and ('USD' in norm or 'ZWG' in norm or 'ZWL' in norm):
                candidates.append(col)
        return candidates

    amount_cols = find_amount_column(df.columns)

    if not amount_cols:
        st.error("No valid Amount column found. Make sure your file has a column containing 'Amount' with USD or ZWL/ZWG.")
        st.stop()

    # Let user select if multiple matches
    if len(amount_cols) > 1:
        currency_column = st.selectbox("Select Amount Column", amount_cols)
    else:
        currency_column = amount_cols[0]

    df['Amount'] = df[currency_column].apply(clean_amount)

    # Validate required columns
    for col in ['Date','Payer','Amount']:
        if col not in df.columns:
            st.error(f"Your file must contain at least these columns: Date, Payer, Amount")
            st.stop()

    df['Date'] = pd.to_datetime(df['Date'])
    df['PAYER_TYPE'] = df['Payer'].apply(categorize_payer)

    # Filters
    st.sidebar.header("Filters")
    medical_aid_options = ['All'] + sorted([aid for aid in MEDICAL_AIDS_DYNAMIC])
    selected_aids = st.sidebar.multiselect("Select Medical Aids to Include", options=medical_aid_options, default=['All'])
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    start_date, end_date = st.sidebar.date_input("Select Date Range", [min_date, max_date])

    df_filtered = df.copy()
    if 'All' not in selected_aids:
        df_filtered = df_filtered[df_filtered['Payer'].str.upper().isin([a.upper() for a in selected_aids])]
    df_filtered = df_filtered[(df_filtered['Date'] >= pd.to_datetime(start_date)) & (df_filtered['Date'] <= pd.to_datetime(end_date))]

    # Overall summary
    total_payments = df_filtered['Amount'].sum()
    total_medical = df_filtered[df_filtered['PAYER_TYPE']=='Medical Aid']['Amount'].sum()
    total_individual = df_filtered[df_filtered['PAYER_TYPE']=='Individual']['Amount'].sum()
    st.subheader("Overall Summary")
    st.write(f"Grand Total Payments: {total_payments:,.2f}")
    st.write(f"Total Medical Aid Payments: {total_medical:,.2f}")
    st.write(f"Total Individual Payments: {total_individual:,.2f}")

    # Medical Aid Analysis
    aid_summary = aggregate_by_aid(df_filtered)
    st.subheader("Top 3 Medical Aids")
    st.dataframe(aid_summary.head(3))

    # Individual Analysis
    individual_summary = aggregate_individuals(df_filtered)
    st.subheader("Top 3 Individual Payers")
    st.dataframe(individual_summary.head(3))

    # Trends
    st.subheader("Monthly Payment Trends")
    medical_trend, individual_trend = calculate_trends(df_filtered)
    fig, ax = plt.subplots()
    ax.plot(medical_trend['Period'].astype(str), medical_trend['Amount'], marker='o', label='Medical Aid')
    ax.plot(individual_trend['Period'].astype(str), individual_trend['Amount'], marker='o', label='Individual')
    ax.set_ylabel("Total Amount")
    ax.set_xlabel("Period")
    ax.set_title("Monthly Payment Trends")
    plt.xticks(rotation=45)
    ax.legend()
    st.pyplot(fig)

    # Month comparison
    if compare_prev_month:
        comparison, current_month, previous_month = generate_comparison(df_filtered)
        st.subheader(f"Month-over-Month Comparison: {previous_month} vs {current_month}")
        st.dataframe(comparison[['Payer','Total_Amount_Previous','Total_Amount_Current','Difference','Percent_Change']])

    # Excel Report
    st.subheader("Download Excel Report")
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_filtered.to_excel(writer, sheet_name="Filtered Transactions", index=False)
        aid_summary.to_excel(writer, sheet_name="Medical Aid Summary", index=False)
        individual_summary.to_excel(writer, sheet_name="Individual Summary", index=False)
        if compare_prev_month:
            comparison.to_excel(writer, sheet_name="Month Comparison", index=False)
        medical_trend.to_excel(writer, sheet_name="Medical Trend", index=False)
        individual_trend.to_excel(writer, sheet_name="Individual Trend", index=False)
        writer.save()
    output.seek(0)
    st.download_button(
        "Download Excel Report",
        data=output,
        file_name=f"Direct_Deposit_Report_{datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.success(f"Detected Amount column: {currency_column}")
else:
    st.info("Upload a Direct Deposit Excel or CSV file to begin analysis.")
