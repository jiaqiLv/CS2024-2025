1. Dependencies**: Make sure you have the required Python packages:

   ```
   pip install numpy scikit-learn
   ```

2. **Running the Program**:

   - To test the program on the Iris dataset, run:

     ```
     python gmm.py
     ```

3. **Parameters**:

   - Modify `args.dataset` in `main()` to switch datasets (options: `"iris"`, `"wine"`).
   - Adjust `n_components` to change the number of Gaussian components in the GMM.