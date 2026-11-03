# NIDS preprocessing analysis

## 1. Split sizes

```
                           group     rows  features  file_MB  percent_of_group
set                                                                           
train_supervised      supervised  1538478        69    182.6             70.70
val_supervised        supervised   319674        69     39.5             14.69
test_supervised       supervised   317953        69     39.4             14.61
train_anomaly_benign     anomaly  1304306        69    151.3             67.17
val_anomaly              anomaly   319674        69     39.5             16.46
test_anomaly             anomaly   317953        69     39.4             16.37
heldout_portscan         holdout     1958        69      0.2            100.00
```

Supervised total: 2,176,105 rows. percent_of_group is the share inside its own group (supervised / anomaly / holdout).

## 2a. Supervised class counts

```
                  train_supervised  val_supervised  test_supervised
Label                                                              
BENIGN                     1304306          269495           267774
DoS Hulk                    120994           25927            25928
DDoS                         89611           19202            19203
DoS GoldenEye                 7197            1542             1543
FTP-Patator                   4153             890              890
DoS slowloris                 3762             805              805
DoS Slowhttptest              3660             784              784
SSH-Patator                   2253             483              483
Web Attack                    1500             322              321
Bot                           1009             216              216
Infiltration                    25               6                5
Heartbleed                       8               2                1
```

Largest gap in a class's share between train/val/test: 0.561 percentage points (near 0 means stratified).

## 2b. Anomaly val/test binary labels (assumed 0 = benign, 1 = attack; confirm)

```
         val_anomaly  test_anomaly
Anomaly                           
0             269495        267774
1              50179         50179
```

## 2c. Held-out set labels

```
         rows
Anomaly      
Bot      1958
```

## 3. Pipeline artifacts

```
                                       value
item                                        
features (no-log1p version)               68
features (log1p version)                  68
columns transformed with log1p            56
twin_drop_index entries                    2
classes in label_mapping                  12
scaler type                     MinMaxScaler
scaler n_features_in_                     68
```

log1p columns (first 15): ['Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow IAT Std', 'Fwd IAT Total', 'Fwd IAT Mean']

## 4. Data quality (both versions)

```
                               NaN  inf  constant_cols  min  max
version  set                                                    
no_log1p train_supervised        0    0              0  0.0  1.0
         val_supervised          0    0              0  0.0  1.0
         test_supervised         0    0              0  0.0  1.0
         train_anomaly_benign    0    0              0  0.0  1.0
         val_anomaly             0    0              0  0.0  1.0
         test_anomaly            0    0              0  0.0  1.0
         heldout_portscan        0    0              8  0.0  1.0
log1p    train_supervised        0    0              0  0.0  1.0
         val_supervised          0    0              0  0.0  1.0
         test_supervised         0    0              0  0.0  1.0
         train_anomaly_benign    0    0              0  0.0  1.0
         val_anomaly             0    0              0  0.0  1.0
         test_anomaly            0    0              0  0.0  1.0
         heldout_portscan        0    0              8  0.0  1.0
```

NaN and inf must be 0. min/max show the value range after scaling.

## 5. log1p effect (no_log1p vs log1p, train_supervised sample)

```
                             skew_before  skew_after    kurt_before     kurt_after  max_abs_z_before  max_abs_z_after  pct_beyond_5_before  pct_beyond_5_after  log1p_applied
CWE Flag Count                547.723022  547.723022  300000.000000  300000.000000             1.000            1.000                  0.0                 0.0           True
Fwd URG Flags                 547.723022  547.723022  300000.000000  300000.000000             1.000            1.000                  0.0                 0.0           True
act_data_pkt_fwd              235.024002    1.286000   56624.281250       3.157000             0.878            0.989                  0.0                 0.0           True
Total Length of Bwd Packets   217.227997   -0.188000   48656.945312      -0.724000             0.957            0.998                  0.0                 0.0           True
Subflow Bwd Bytes             217.222000   -0.188000   48656.355469      -0.724000             0.957            0.998                  0.0                 0.0           True
Subflow Bwd Packets           215.126999    1.301000   47540.500000       3.473000             0.910            0.992                  0.0                 0.0           True
Total Backward Packets        215.126999    1.301000   47540.500000       3.473000             0.910            0.992                  0.0                 0.0           True
Total Fwd Packets             214.688004    1.494000   47235.296875       3.668000             0.899            0.991                  0.0                 0.0           True
Subflow Fwd Packets           214.688004    1.494000   47235.296875       3.668000             0.899            0.991                  0.0                 0.0           True
Total Length of Fwd Packets   135.931000   -0.282000   25684.658203       0.252000             0.142            0.881                  0.0                 0.0           True
Subflow Fwd Bytes             135.931000   -0.282000   25684.658203       0.252000             0.142            0.881                  0.0                 0.0           True
RST Flag Count                 57.077999   57.077999    3255.924072    3255.924072             1.000            1.000                  0.0                 0.0           True
ECE Flag Count                 57.077999   57.077999    3255.924072    3255.924072             1.000            1.000                  0.0                 0.0           True
Active Min                     46.838001    1.404000    5230.157227       0.273000             0.950            0.997                  0.0                 0.0           True
Active Std                     41.549999    2.933000    4530.138184       6.849000             0.950            0.997                  0.0                 0.0           True
```

Skewness is unchanged by linear scaling, so comparing the two scaled versions is valid.
Features: 68; log1p applied to 56.
On the log1p columns: median |skew| 6.84 -> 0.74; median kurtosis 73.9 -> -0.5; median max |z| 1.0 -> 1.0; median % of values beyond 5 std 0.000 -> 0.000.
Columns NOT in log1p_columns identical between versions (sanity check): True.

## 6. Leakage checks

```
                                                                    value
check                                                                    
duplicate rows inside train_supervised                              24733
duplicate rows inside val_supervised                                  901
duplicate rows inside test_supervised                                 834
rows of val_supervised also present in train_supervised                80
rows of test_supervised also present in train_supervised               84
rows of test_supervised also present in val_supervised                 20
holdout labels ['Bot'] present in train                                 1
train_supervised: mean |column mean| / mean column std    0.2601 / 0.2093
val_supervised: mean |column mean| / mean column std      0.2624 / 0.2107
test_supervised: mean |column mean| / mean column std     0.2626 / 0.2112
```

A scaler fitted on train only gives train mean ~0 and std ~1, with val/test close but not exact. Identical rows across splits are normal in CICIDS2017 (repeated flows); report the count and, if large, mention it as a limitation. The holdout label must NOT appear in train (value 0).

## 7. Parquet vs CSV (first 500k rows of X_train_supervised)

```
                          csv  parquet  csv_over_parquet
file_size_MB          281.372   60.206              4.67
write_time_s           25.869    2.357             10.98
read_all_columns_s      4.258    0.150             28.39
read_10_columns_s       2.433    0.041             59.34
memory_after_load_MB  272.000  140.000              1.94
```

Parquet round-trips dtypes exactly: True (CSV reloads as float64 and loses float32).

## Reasons to cite in the report (edit to match what you did)

**Why separate supervised and anomaly datasets:** Random Forest and XGBoost need labelled
data from all classes (supervised sets). The Autoencoder and Isolation Forest learn what
normal traffic looks like, so their training set (X_train_anomaly_benign) contains benign
flows only, while their val/test sets contain benign and attack flows with binary labels.

**Why train/val/test:** train fits the models, validation is used for tuning and threshold
selection, test is used once for the final score. The split is stratified (see
class_percent_supervised.csv), so rare classes appear in every split.

**Why a PortScan holdout:** PortScan is excluded from training and evaluated separately
to measure detection of an attack type the models have never seen.

**Why log1p:** flow features (durations, byte and packet counts, inter-arrival times) are
heavily right-skewed with a few huge values. log(1+x) compresses the tail, preserves
ordering, and is defined at 0. After log1p the skewness, kurtosis and extreme z-scores
drop (see figures 03-05), which helps the Autoencoder and Isolation Forest. Tree models
(Random Forest, XGBoost) are invariant to monotonic transforms, so it does not change them.

**Why scaling fitted on train only:** prevents information from val/test leaking into
training; the scaler is saved (scaler.joblib) so the same transform is applied at inference.

**Why Parquet:** compressed columnar format, so files are smaller and load faster than CSV,
dtypes are preserved with no re-parsing, and single columns can be read without loading the
whole file (see figure 06 and parquet_vs_csv.csv).
