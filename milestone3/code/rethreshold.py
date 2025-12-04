import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_csv")
parser.add_argument("--threshold", type=float)
parser.add_argument("--output_csv")
args = parser.parse_args()

df = pd.read_csv(args.input_csv)
df["y_pred"] = (df["y_prob"] >= args.threshold).astype(int)
df.to_csv(args.output_csv, index=False)
print("Saved:", args.output_csv)
