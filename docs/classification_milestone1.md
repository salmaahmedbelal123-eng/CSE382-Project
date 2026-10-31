# Classification (IsCanceled): Milestone 1 section

**Prepared by:** Lujy  |  **Task:** predict whether a booking will eventually be canceled

## 1. Prediction point
We assume the prediction is made **at the moment the booking is created**, before anything else
happens to it (no check-in, no cancellation, no later changes). A column may only be used as an
input if its value was already known at that moment.

## 2. Feature choices
**Removed (leakage risk):**

| Column | Why it leaks |
| --- | --- |
| ReservationStatus | Values are Check-Out / Canceled / No-Show. It is the answer written in words (direct leakage). |
| ReservationStatusDate | The date this final status was set. Tied to ReservationStatus (direct leakage). |
| AssignedRoomType | Assigned at check-in. A canceled booking never reaches check-in (indirect leakage). |
| BookingChanges | Counts changes "before check-in or cancellation", so it keeps growing after booking time (indirect). |
| RequiredCarParkingSpaces | Only guests who actually arrive end up with parking, so canceled bookings are almost always 0 (indirect). |
| TotalOfSpecialRequests | Requests are added over time and the dataset stores the final count (indirect). |

**Kept (known at booking time):** LeadTime, ArrivalDateYear / Month / WeekNumber / DayOfMonth,
StaysInWeekendNights, StaysInWeekNights, Adults, Children, Babies, Meal, Country, MarketSegment,
DistributionChannel, IsRepeatedGuest, PreviousCancellations, PreviousBookingsNotCanceled,
ReservedRoomType, DepositType, Agent, DaysInWaitingList, CustomerType, ADR, has_company.

**Borderline, kept on purpose (to confirm with the team):**
- ADR: stored as the final average daily rate, but it comes from the rate plan chosen at booking.
- DaysInWaitingList: the waiting list ends when the booking is confirmed, so it is known early.
- DepositType: known at booking. "Non Refund" is strongly linked to cancellation in this data. That is a real pattern, not leakage.

## 3. Preprocessing (inside one scikit-learn Pipeline)
- Numeric columns: missing values filled with the training mean, then standardized.
- Text columns: missing values filled with the most frequent training value, then one-hot encoded
  (categories under 1% of rows are merged into one "infrequent" column; unseen categories do not crash).
- Agent: already one-hot encoded by the team's preprocessing step (missing = booking without an agent, kept as its own category `NoAgent`).
- Same split as the team (stratified 70/15/15, seed 42, files in `data/processed/`).

## 4. Preliminary leakage audit (classification)
- **Direct leakage:** ReservationStatus and ReservationStatusDate are removed.
- **Indirect leakage:** AssignedRoomType, BookingChanges, RequiredCarParkingSpaces and TotalOfSpecialRequests
  are removed because their stored values are only final after the booking has progressed.
- **Preprocessing leakage:** filling, scaling and encoding are learned on training rows only, because they sit inside the Pipeline.
- **Test set:** not used in Milestone 1. It stays untouched for the final evaluation.
- Still to review together: the borderline columns above.

## 5. Initial experiment
Two untuned models (Logistic Regression, Random Forest), default settings, scored on the validation set
with accuracy, precision, recall, F1, ROC-AUC, PR-AUC and the confusion matrix.
Command: `python -m src.classification.train`. Results: `outputs/classification/baseline_results.csv`.

## 6. Plan for the remaining classification work
1. Add Decision Tree and XGBoost, so there are at least 3 models.
2. Hyperparameter tuning for every model (random search, 3-fold cross-validation on the training set, scored by F1).
3. Baseline vs tuned comparison table, and a stated rule to choose the deployed model (validation set only).
4. PCA and LDA: compare original features vs PCA vs LDA features inside the same Pipeline, together with the team.
5. One final evaluation on the test set.
6. Save the final pipeline and add a prediction function for the application.

## 7. Initial results (validation set, untuned models)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Always guess "not canceled" | 0.698 | - | 0 | - | - |
| Logistic Regression | 0.721 | 0.565 | 0.329 | 0.416 | 0.776 |
| Random Forest | 0.774 | 0.656 | 0.531 | 0.587 | 0.842 |

Full table with confusion matrices: `outputs/classification/baseline_results.csv`.

## 8. Interpretation
- About 30% of bookings are canceled, so always guessing "not canceled" already gives 69.8% accuracy. Accuracy alone is misleading, so we focus on recall and F1.
- Logistic Regression is only slightly above that baseline (72.1%) and finds only 32.9% of the real cancellations, so it misses about two thirds of them.
- Random Forest is better on every metric (accuracy 77.4%, recall 53.1%, F1 0.587, ROC-AUC 0.842). A likely reason is that it can capture non-linear patterns and interactions between features, which a linear model cannot.
- Both models are untuned, so these are starting points. Recall is still low, so in Milestone 2 we will tune the models and try class weighting to catch more cancellations.
- The test set was not used.
