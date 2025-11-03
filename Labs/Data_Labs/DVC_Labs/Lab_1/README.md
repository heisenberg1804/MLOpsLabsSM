# DVC Data Versioning Lab: The "Time Machine"

## 1. Objective

This project demonstrates the core power of Data Version Control (DVC) when used with Git. The goal is to show how to version data like code, allowing us to:
* Track different versions of a dataset without storing large files in Git.
* Use Git commits as "pointers" to specific dataset versions.
* Switch between these data versions seamlessly, creating a "time machine" for our data.

This lab uses a GCS bucket (`gs://dvc-lab-bucket-mlops`) as remote storage.

## 2. The Versioning Workflow (What We Did)

We created two distinct versions of our dataset and pushed them to our remote storage.

### Version 1: The Original Data

1.  **Track:** The original `data/CC_GENERAL.csv` file was added to DVC.
    ```bash
    dvc add data/CC_GENERAL.csv
    ```
2.  **Commit:** The resulting "pointer file" (`data/CC_GENERAL.csv.dvc`) was committed to Git.
    ```bash
    git add data/CC_GENERAL.csv.dvc data/.gitignore
    git commit -m "feat: Track v1 (original) of CC_GENERAL.csv"
    ```
3.  **Push:** The actual large data file was pushed to the GCS remote.
    ```bash
    dvc push
    ```
**Result:** The first Git commit now points to the *original* dataset.

### Version 2: The Modified Data

1.  **Modify:** The `data/CC_GENERAL.csv` file was intentionally modified (e.g., by removing the first 10 rows) to create a new version.
2.  **Track:** `dvc add` was run again, detecting the change and generating a new hash.
    ```bash
    dvc add data/CC_GENERAL.csv
    ```
3.  **Commit:** The *updated* pointer file (`data/CC_GENERAL.csv.dvc`) was committed to Git.
    ```bash
    git add data/CC_GENERAL.csv.dvc
    git commit -m "feat: Track v2 (modified) of CC_GENERAL.csv"
    ```
4.  **Push:** The new data file was pushed to GCS. DVC was smart enough to only upload the new version, leaving the original data untouched.
    ```bash
    dvc push
    ```
**Result:** The `main` branch now points to the *modified* dataset.

## 3. 🚀 The "Time Machine" Demonstration

This is how to switch between the two data versions we created.

> **Note:** These commands use `dvc checkout`, which pulls data from the local DVC cache. If you are on a new machine, you would run `dvc pull` to download the data from GCS first.

### Step 1: Check the Current Data (v2)

By default, we are on the `main` branch, which points to our latest work (v2).

```bash
# 1. Ensure you are on the main branch
git checkout main

# 2. Sync data with the current Git commit
dvc checkout
```
At this point, if you inspect `data/CC_GENERAL.csv`, you will see the **modified file** (v2).

### Step 2: Travel Back in Time (v1)

Now, let's go back in time to the *first* data commit.

```bash
# 1. Use Git to check out the previous commit
#    (HEAD~1 means "one commit before the most recent one")
git checkout HEAD~1

# 2. Sync data with this older Git commit
dvc checkout
```
**Magic!** If you inspect `data/CC_GENERAL.csv` now, you will see the **original, unmodified file** (v1), magically restored.

### Step 3: Return to the Present (v2)

Finally, let's return to the present.

```bash
# 1. Go back to the main branch
git checkout main

# 2. Sync data with the main branch
dvc checkout
```
The `data/CC_GENERAL.csv` file is once again the **modified version** (v2).

## 4. Conclusion

This workflow proves that DVC + Git allows for complete, reproducible versioning of a machine learning project. We can reliably tie our code, data, and models together at any given point in history.