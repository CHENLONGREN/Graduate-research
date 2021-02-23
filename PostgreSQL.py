import psycopg2
import pandas as pd
from sqlalchemy import create_engine


conn = psycopg2.connect(database='dev', user='sophia', password='jg5Ac_E4', host='jointresearch.c4djdhnq7lau.ap-northeast-1.redshift.amazonaws.com', port='5439')
print("Opened database successfully")

cur = conn.cursor()

# cur.execute("select rp_date, cp_id, imps, clicks, cost,cv_t,cv_u from google_cp where cp_id = '652057083' ORDER BY rp_date LIMIT 3009;")
# print("Operation done successfully")
# conn.close()

# df = pd.read_sql_query("select rp_date, cp_id, device, imps, clicks, cost,cv_t from google_cp where cp_id='652057083' AND device=1  ORDER BY rp_date;",
#                        con=create_engine('postgresql://sophia:jg5Ac_E4@jointresearch.c4djdhnq7lau.ap-northeast-1.redshift.amazonaws.com:5439/dev'))
df = pd.read_sql_query("select rp_date, cp_id, device, imps, clicks, cost,cv_t from google_cp where cp_id='622861879' ORDER BY rp_date;",
                       con=create_engine('postgresql://sophia:jg5Ac_E4@jointresearch.c4djdhnq7lau.ap-northeast-1.redshift.amazonaws.com:5439/dev'))
df.to_csv(r'D:\Data\cp622861879.csv')

