import textwrap
import pandas as pd

from cvdms_platform import CvdmsApp

def wrap_cell(s, width=60):
    if s is None:
        return ""
    s = str(s)
    return "\n".join(textwrap.wrap(s, width=width)) if len(s) > width else s

def pretty_print_wrapped(df, width=60, max_rows=50):
    df2 = df.copy()
    for col in df2.select_dtypes(include=["object", "string"]).columns:
        df2[col] = df2[col].apply(lambda v: wrap_cell(v, width))
    print(df2.head(max_rows).to_string(index=False))

app = CvdmsApp(app_name="cvdmsv1",
               profile_name="developers_admin")

upload_ok, upload_attempt_info = app.upload_imagery("sample/sample.csv",
                                                     summary="my first test with the new infrastructure",
                                                     data_source='FashionMNIST')

job_id = upload_attempt_info.get("job_id")
log_retrieval_success, log_info, log_df_results = app.get_logs_by_job_id(job_id)

# log_df_results is a df you can save if wanted, but note it takes time for logs to show up.
# log_df_results.to_csv(f'logs_{job_id}.csv', index=False)

# Pretty print the logs so far. Re-run get_logs_by_job_id above to update logs.
pretty_print_wrapped(log_df_results, width = 60, max_rows = 600)