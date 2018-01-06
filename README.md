## 操作流程

1. 前往[本網頁](https://datashare.is.ed.ac.uk/handle/10283/2791)下載資料集，請下載下列項目並解壓縮。
*  `clean_trainset_28spk_wav.zip`
*  `noisy_trainset_28spk_wav.zip`
*  `clean_testset_wav.zip`
*  `noisy_testset_wav.zip`

2. 對原始檔案進行重新取樣爲 22050 取樣率，可以使用 `resample_wav.sh` 腳本。

```sh
# 重新取樣後儲存至 resampled_{clean,noisy}
./resample_wav.sh 22050 clean_trainset_28spk_wav/ resampled_clean/
./resample_wav.sh 22050 noisy_trainset_28spk_wav/ resampled_noisy/
```

3. 使用 `pack_dataset.py` 打包資料集，如下範例會生成 `dataset_clean.bin` 、 `dataset_noisy.bin` 、 `dataset_filenames.txt` 等檔案。

```sh
# 打包爲取樣率 22050 、 長度 4 秒的資料集，儲存輸出檔案路徑皆有 dataset_ 前綴
./pack_dataset.py 22050 4 resampled_clean/ resampled_noisy dataset_
```

4. 使用 `main.py` 執行訓練。
TODO
