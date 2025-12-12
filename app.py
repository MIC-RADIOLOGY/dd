import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
from datetime import datetime

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
    # Load file
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    # Map sample DD columns
    df = df.rename(columns={
        'Transaction date': 'Date',
        'Transaction description': 'Payer'
    })

    # Let user select which currency to analyze
    currency_column = st.sidebar.selectbox("Select Amount Currency", options=['Amount (USD)', 'Amount (ZWG)'])
    df['Amount'] = df[currency_column]

    # Optional: keep Receipt Number
    if 'Receipt Number' in df.columns:
        df['Receipt Number'] = df['Receipt Number']

    # Validate required columns
    required_cols = ['Date','Payer','Amount']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Your file must contain at least these columns: {', '.join(required_cols)}")
        st.stop()

    # Clean data
    df['Date'] = pd.to_datetime(df['Date'])
    df['Amount'] = df['Amount'].apply(clean_amount)
    df['PAYER_TYPE'] = df['Payer'].apply(categorize_payer)

    # ---------- Filters ----------
    st.sidebar.header("Filters")
    medical_aid_options = ['All'] + sorted([aid for aid in MEDICAL_AIDS_DYNAMIC])
    selected_aids = st.sidebar.multiselect("Select Medical Aids to Include", options=medical_aid_options, default=['All'])
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    start_date, end_date = st.sidebar.date_input("Select Date Range", [min_date, max_date])

    # Apply filters
    df_filtered = df.copy()
    if 'All' not in selected_aids:
        df_filtered = df_filtered[df_filtered['Payer'].str.upper().isin([a.upper() for a in selected_aids])]
    df_filtered = df_filtered[(df_filtered['Date'] >= pd.to_datetime(start_date)) & (df_filtered['Date'] <= pd.to_datetime(end_date))]

    # ---------- Overall Summary ----------
    total_payments = df_filtered['Amount'].sum()
    total_medical = df_filtered[df_filtered['PAYER_TYPE']=='Medical Aid']['Amount'].sum()
    total_individual = df_filtered[df_filtered['PAYER_TYPE']=='Individual']['Amount'].sum()
    st.subheader("Overall Summary")
    st.write(f"**Grand Total Payments:** {total_payments:,.2f}")
    st.write(f"**Total Medical Aid Payments:** {total_medical:,.2f}")
    st.write(f"**Total Individual Payments:** {total_individual:,.2f}")

    # ---------- Medical Aid Analysis ----------
    aid_summary = aggregate_by_aid(df_filtered)
    top3_aid = aid_summary.head(3)
    st.subheader("Top 3 Medical Aids")
    st.dataframe(top3_aid)

    st.subheader("Top 10 Medical Aids by Payment")
    top10_aid = aid_summary.head(10)
    fig, ax = plt.subplots()
    ax.bar(top10_aid['Payer'], top10_aid['Total_Amount'])
    ax.set_ylabel("Total Amount")
    ax.set_xlabel("Medical Aid")
    ax.set_title("Top 10 Medical Aids")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # ---------- Individual Payer Analysis ----------
    individual_summary = aggregate_individuals(df_filtered)
    top3_individuals = individual_summary.head(3)
    st.subheader("Top 3 Individual Payers")
    st.dataframe(top3_individuals)

    st.subheader("Top 10 Individual Payers by Payment")
    top10_individuals = individual_summary.head(10)
    fig_ind, ax_ind = plt.subplots()
    ax_ind.bar(top10_individuals['Payer'], top10_individuals['Total_Amount'], color='orange')
    ax_ind.set_ylabel("Total Amount")
    ax_ind.set_xlabel("Individual Payer")
    ax_ind.set_title("Top 10 Individual Payers")
    plt.xticks(rotation=45)
    st.pyplot(fig_ind)

    # ---------- Month Comparison ----------
    if compare_prev_month:
        comparison, current_month, previous_month = generate_comparison(df_filtered)
        st.subheader(f"Month-over-Month Comparison: {previous_month} vs {current_month}")
        st.dataframe(comparison[['Payer','Total_Amount_Previous','Total_Amount_Current','Difference','Percent_Change']])

        st.subheader("Top 5 Medical Aids: Month Comparison")
        top5 = comparison.head(5)
        fig2, ax2 = plt.subplots()
        ax2.bar(top5['Payer'], top5['Total_Amount_Previous'], label=str(previous_month))
        ax2.bar(top5['Payer'], top5['Total_Amount_Current'], alpha=0.7, label=str(current_month))
        ax2.set_ylabel("Total Amount")
        ax2.set_xlabel("Medical Aid")
        ax2.set_title("Previous vs Current Month Payments")
        plt.xticks(rotation=45)
        ax2.legend()
        st.pyplot(fig2)

    # ---------- Trends & Growth ----------
    st.subheader("Monthly Payment Trends")
    medical_trend, individual_trend = calculate_trends(df_filtered, freq='M')
    fig3, ax3 = plt.subplots()
    ax3.plot(medical_trend['Period'].astype(str), medical_trend['Amount'], marker='o', label='Medical Aid')
    ax3.plot(individual_trend['Period'].astype(str), individual_trend['Amount'], marker='o', label='Individual')
    ax3.set_ylabel("Total Amount")
    ax3.set_xlabel("Period")
    ax3.set_title("Monthly Payment Trends")
    plt.xticks(rotation=45)
    ax3.legend()
    st.pyplot(fig3)

    st.subheader("Monthly Growth Percentages")
    growth_df = pd.DataFrame({
        "Period": medical_trend['Period'].astype(str),
        "Medical_Aid_Growth_%": medical_trend['Growth_%'],
        "Individual_Growth_%": individual_trend['Growth_%']
    })
    st.dataframe(growth_df)

    # ---------- Excel Report ----------
    st.subheader("Download Excel Report")
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_filtered.to_excel(writer, sheet_name="Filtered Transactions", index=False)
        aid_summary.to_excel(writer, sheet_name="Medical Aid Summary", index=False)
        top3_aid.to_excel(writer, sheet_name="Top 3 Medical Aids", index=False)
        individual_summary.to_excel(writer, sheet_name="Individual Summary", index=False)
        top3_individuals.to_excel(writer, sheet_name="Top 3 Individuals", index=False)
        if compare_prev_month:
            comparison.to_excel(writer, sheet_name="Month Comparison", index=False)
        medical_trend.to_excel(writer, sheet_name="Medical Trend", index=False)
        individual_trend.to_excel(writer, sheet_name="Individual Trend", index=False)
        growth_df.to_excel(writer, sheet_name="Growth %", index=False)
        writer.save()
    output.seek(0)
    st.download_button(
        "Download Excel Report",
        data=output,
        file_name=f"Direct_Deposit_Report_{datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("Upload a Direct Deposit Excel or CSV file to begin analysis.")
