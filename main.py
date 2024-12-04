from __future__ import annotations

# cr241526

import polars as pl
from polars import read_csv, scan_csv
import altair as alt

# a) Import the data from the “doctors_fee.csv” file into a dataframe.

df = read_csv(
    "doctors_fees-v2.csv",
    separator=";",
    decimal_comma=True,
    schema={
        "ID": pl.UInt32,
        "Age": pl.UInt8,
        "Sex": pl.Categorical,
        "BMI": pl.Float32,
        "Children": pl.UInt8,
        "Smoker": pl.String,
        "Region": pl.Categorical,
        "Charges": pl.Float64,
    },
).with_columns(pl.col("Smoker") == "yes")


# b) How many rows and how many columns does the data frame have?

print(df.shape)

# c) Delete the column id from the dataframe.

ids = df.drop_in_place("ID")

# d) Change the column names to lower case.

df = df.rename(str.lower)

# e) Change the column names “sex” to “gender”.

df = df.rename({"sex": "gender"})

# f) In which columns are there how many missing values?

print(df.null_count())

# g) If there are missing values, delete the corresponding rows from the dataframe.

df = df.drop_nulls()
print(df.shape)

# h) Replace „female“ with 0, and „Male“ with 1.

df = df.with_columns((pl.col("gender") == "male").cast(pl.UInt8))

# i) Have you found incorrect values (which ones)?

print(df.filter(pl.col("gender") > 1))

# j) Estimate the distribution of age with a visualization. Comment on this distribution.

df.get_column("age").plot.hist()

# k) What is the distribution of gender? What is the distribution on smoker? What is the
# distribution on region? Comment these distributions.

df.get_column("gender").plot.hist()
df.get_column("smoker").plot.hist()
df.get_column("region").plot.hist()

# l) Estimate the distribution of “bmi” with a visualization. Comment on this distribution

df.get_column("bmi").plot.hist()

# m) What is the distribution of the charges?

df.get_column("charges").plot.hist()

# n) Create a correlation matrix (with the numeric variables including gender). Comment all
# correlation values. E.g.: on average, do women or mean have higher charges? Which
# correlations look strange?

df.select(pl.selectors.numeric()).corr()

# Create a scatterplot with „charges“ vs age. Comment this plot.

df.plot.scatter(x="age", y="charges")

# p) Create a „side-by-side-boxplot“ of your choice. Comment this plot.

boxplot = (
    alt.Chart(df)
    .mark_boxplot()
    .encode(
        x=alt.X("age", axis=alt.Axis(title="age")),
        y=alt.Y("charges", axis=alt.Axis(title="charges")),
        color=alt.Color("gender", legend=alt.Legend(title="gender")),
    )
)

boxplot.show()

# q) Create a new column „bmi_age“ – as the multiplication of bmi and age.

df = df.with_columns(pl.col("bmi") * pl.col("age"))


def main() -> int:
    print(df.head())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
