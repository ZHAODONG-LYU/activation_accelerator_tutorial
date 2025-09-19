# Test vector

To effectively evaluate the FPGA hardware acceleration module's floating-point computation accuracy and special value handling capabilities for seven core activation functions (**Softmax**, **LayerNorm**, **RMSNorm**, **SiLU**, **GELU**, **Elementwise Add**, **Elementwise Multiply**) under the bf16 data type, a comprehensive and hierarchical testing strategy needs to be designed. This strategy should cover functional correctness, precision boundaries, special value scenarios, edge case scenarios, and simulation of real-world data distributions.

## Test Strategy Overview

### Purpose

1. Accuracy Verification:  Ensure that the FPGA-implemented bf16 floating-point operation results are highly consistent with the PyTorch bf16 reference results, meeting the criteria in the scoring standard $\epsilon_f \leq 10^{-3}   $.
2. Special Value Handling: Verify that the hardware module correctly processes special values in bf16 floating-point numbers (`+0.0, -0.0, +Inf, -Inf, NaN`, Subnormal number)
3. Boundary Conditions: Test the behavior when input data is at the boundaries of the bf16 representation range (maximum/minimum normal numbers, subnormal numbers).
4. Functional Completeness: Ensure that all seven activation functions work correctly under various valid inputs.
5. Batch Processing: Verify the correct handling of `N x Ddimensional tensors (N=4, D = 768)`, including parallelism and data flow.

### bf16 data type characteristics

bf16 (bfloat16) has 1 sign bit, 8 exponent bits, and 7 mantissa bits. Its range is similar to float32, but with lower precision. During testing, special attention should be paid to its rounding behavior (typically round-to-nearest-even), the handling of subnormal numbers (whether they are flushed to zero or computed precisely), and the propagation rules for special values (`Inf, NaN`).

## Test Vector Classification and Design

The test vectors will be categorized into the following types to ensure comprehensive coverage:

### A. Basic functionality correctness testing

#### Purpose

Verify that the core computation logic of each activation function is correctly implemented according to its mathematical definition.

#### General Prinpicle

Use simple, predictable inputs whose output results are easy to verify manually or through simple calculations.

#### Softmax

* All-Zero input: `[0.0, 0.0, ..., 0.0]`. The output should be $[\frac{1}{D}, ..., \frac{1}{D}]$.
* A row with a single element as 1 and the rest as 0: For example, `[0.0, 1.0, 0.0, ...]`. Verify the exponential normalization and the "subtract the maximum" logic for overflow prevention.
* Rows with all identical elements: For example, `[2.5, 2.5, ..., 2.5]`. The output should be $[\frac{1}{D}, ..., \frac{1}{D}]$.
* Cases where input values vary significantly: For example, `[0.0, 0.0, 10.0, 0.0]` or `[100.0, 101.0, 102.0]`. Test the stability of exp calculation and max subtraction operations to ensure the output sums to 1.

#### LayerNorm/RMSNorm

* Input with a mean of 0 and a variance of 1: For example, `[1.0, -1.0, 1.0, -1.0, ...]`
* Rows with all identical elements: For example, `[5.0, 5.0, ...,5.0]`. Verify the accuracy of mean/sum of squares calculation, where variance/root mean square should be close to or equal to 0, testing epsilon's protection against division by zero.
* • Input with zero mean/RMS close to zero: For example, `[0.0, 0.0, ..., 0.0]`

#### SiLU

* Input is `0`: `SiLU(0) = 0`.
* Input is `1.0`: `SiLU(1.0) = 1.0*Sigmoid(1.0) ≈ 0.731 `
* Input is `-5.0`: `SiLU(-5.0) = -5.0 * Sigmoid(-5.0) ≈ -0.0335 `

#### GELU

* Input is `0`: `GeLU(0) = 0`.
* Input is `1.0`: `SiLU(1.0)  ≈ 0.841 `
* Input is `-3.0`: `SiLU(-3.0)  ≈ -0.004`

#### Elementwise Add/Mul

* Add: Input `1` is all `1`s, input `2` is all `2`s. The result should be all `3`s.
* Mul: Input `1` is all `1`s, input `2` is all `2`s. The result should be all `2`s.
* Mul: One of Input is all `0`s, the result should be all `0`s.

### B. Precision-sensitive area test

#### Purpose

Verify that the hardware module can still maintain high accuracy in regions where the function behavior changes drastically and is prone to precision loss.

#### Input with known analytical solution

For certain functions, there are precise mathematical results under specific inputs (for example, SiLU outputs 0 at `x=0`, and LayerNorm outputs `0` when all inputs are equal). Use these points to verify accuracy.

#### Dense sampling in sensitive areas

* Softmax: Inputs are close but slightly different, e.g., `[1.000, 1.001, 1.002]`.
* LayerNorm/RMSNorm: Variance/RMS is very small but non-zero, e.g., input values are concentrated in a very small range.
* SiLU/GELU: nputs are close to `0` (e.g., densely sampled in the range `[-0.1, 0.1]`), as the derivatives of these functions change significantly near `0`.
* Add/Mul: Adding a large number to a small one (e.g., `1000.0 + 0.001`), adding a positive and negative number to get a result close to zero (e.g., `1.00001 + (-1.0)`), multiplying two very small numbers (which may lead to underflow).
* Random Inputs (Large-Scale Accuracy Verification)

  * Generate a large number ofrandom bf16 numbers with uniform distribution in different numerical ranges  (e.g., `[-10.0, 10.0], [-100.0, 100.0], [-1.0, 1.0]`) as inputs.

### C. Special Value and Boundary Value Tests

#### Purpose

To ensure the hardware module correctly handles various special cases and representation limits of bf16 floating-point numbers, following the IEEE 754 (or bf16 equivalent) standard.

#### Special Value Inputs:

* `+Inf` (positive infinity)
* `-Inf` (negative infinity)
* `NaN` (not a number, including quiet and signaling `NaN`, verifing its propagation behavior)
* `+0.0` and `-0.0` (positive zero and negative zero)
* Subnormal numbers: For example, the value of $1e^{-40}$ converted to bf16.

#### Boundary Value Inputs:

* Maximum positive number (`BF16_MAX`)
* Smallest positive normal number (`BF16_MIN_NORMAL`)
* Maximum negative number (`-BF16_MAX`)
* Smallest negative normal number (`-BF16_MIN_NORMAL`)
* Maximum and minimum subnormal numbers.

#### Mixed Inputs:

Create tensors containing a mix of normal, special, and boundary values to test the behavior of the computation unit under complex conditions.

#### Special Value Handing by Function

* Softmax: Inputs containing `+Inf` (one or more), `-Inf`, or `NaN`. For example, `[0.0, Inf, 0.0], [Inf, Inf, 0.0], [NaN , 1.0, 2.0]`.
* LayerNorm/RMSNorm: Inputs containing `Inf` or `NaN`, verifying the propagation of `NaN` in mean/variance/RMS calculations.
* SiLU/GELU: Inputs containing `Inf`, `-Inf`, or `NaN`.
* Add/Mul:
  * `Inf + (-Inf)` (should result in `NaN`)
  * `0 * Inf` (should result in `NaN`)
  * `x + NaN = NaN, x * NaN = NaN`
  * Adding/multiplying two very large numbers (verifying overflow).
  * Multiplying two very small numbers (verifying underflow or subnormal numbers).

### D. Actual Distribution Simulation Tests

#### Purpose

To simulate the actual input data distributions that may occur during large language model inference, verifying the hardware's performance under realistic workloads.

#### Real Data Sampling

If possible, capture intermediate activation values from actual large language model inference processes as test inputs.

#### Statistical Distribution Simulation

Generate random numbers that conform to specific statistical distributions (e.g., normal, uniform, truncated normal distributions), and adjust their mean and variance to simulate the characteristics of activation outputs from different Transformer layers (such as attention output, FFN output).

Based on the proposed test strategy and the fixed input tensor size of `N=64, D=768`, we can design a single `64 x 768` bf16 input tensor (denoted as X_test_tensor), where each row X_test_tensor `[i, :]` represents a specific test case or scenario. For two-input operations like Elementwise Add and Elementwise Multiply, X_test_tensor will serve as the first operand, and the test script will be responsible for generating the corresponding second operand Y.

The following is a detailed filling strategy for X_test_tensor, designed to cover functional correctness, accuracy boundaries, special value scenarios, and actual distribution simulation:

#### X_test_tensor (64 x 768, bf16) Filling Strategy

We will divide the 64 rows of X_test_tensor into the following categories and detail the data content of each row.

1. General BF16 Characteristics and Common Value Tests (12 rows)
   This category of tests is designed to verify the floating-point unit's ability to handle basic numerical and special values, applicable to all activation functions.

   * Row 0: All zeros

     * Content: All elements are `+0.0`.
     * Purpose: Verify behavior with zero input.
   * Row 1: All Ones

     * Content: All elements are `1.0`.
     * Purpose: Verify behavior with unit input.
   * Row 2: All Negative Ones

     * Content: All elements are `-1.0`.
     * Purpose: Verify behavior with negative unit input.
   * Row 3: All `BF16_MAX`

     * Content: All elements are the maximum positive bf16 number.
     * Purpose: Verify behavior with maximum value input, checking overflow boundaries.
   * Row 4: All `-BF16_MAX`

     * Content: All elements are the maximum negative bf16 number.
     * Purpose: Verify behavior with minimum value input, checking overflow boundaries.
   * Row 5: All `BF16_MIN_NORMAL`

     * Content: All elements are the smallest positive normal bf16 number.
     * Purpose: Verify behavior with small positive normal number input.
   * Row 6: All `-BF16_MIN_NORMAL`

     * Content: All elements are the smallest negative normal bf16 number.
     * Purpose: Verify behavior with small negative normal number input.
   * Row 7: All Positive Subnormals

     * Content: All elements are a large positive subnormal number (e.g., a float32 of `1e-30` converted to bf16).
     * Purpose: Verify subnormal number handling.
   * Row 8: All Negative Subnormals

     * Content: All elements are a large negative subnormal number.
     * Purpose: Verify subnormal number handling.
   * Row 9: Mixed Normal Values

     * Content: A repeating pattern of normal values like `[1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.1, -0.1, 10.0, -10.0]`.
     * Purpose: Verify general computation with mixed values.
   * Row 10: Mixed Special Values

     * Content: A repeating pattern of special values and normal values like `[+Inf, -Inf, NaN, +0.0, -0.0, 1.0, -1.0]`.
     * Purpose: Verify the propagation and interaction of special values.
   * Row 11: Alternating `+0.0` and `-0.0`

     * Content: Even indices are `+0.0`, odd indices are `-0.0`.
     * Purpose: Verify handling of signed zeros.
2. Random Distribution Input Tests (8 rows)
   This category simulates actual data distributions and verifies the hardware's accuracy and stability across different numerical ranges.

   * Row 12: Uniform Random `[-10.0, 10.0]`
     * Content: Random numbers uniformly distributed in the range `[-10.0, 10.0]`.
     * Purpose: Common input range for activation functions.
   * Row 13: Uniform Random `[-1000.0, 1000.0]`
     * Content: Random numbers uniformly distributed in the range `[-1000.0, 1000.0]`.
     * Purpose: Test for a larger numerical range.
   * Row 14: Uniform Random `[-0.1, 0.1]`
     * Content: Random numbers uniformly distributed in the range `[-0.1, 0.1]`.
     * Purpose: Sensitive to accuracy for SiLU and GELU near zero.
   * Row 15: Uniform Random `[0.0, 1.0]`
     * Content: Random numbers uniformly distributed in the range `[0.0, 1.0]`.
     * Purpose: All-positive input test.
   * Row 16: Normal Distribution (mean 0, std dev 5)
     * Content: Random numbers from a standard normal distribution, multiplied by 5.
     * Purpose: Simulates common neural network weight or activation distributions.
   * Row 17: Random Values with Injected `Inf/NaN`
     * Content: Mostly random numbers in the range `[-5.0, 5.0]`, with `+Inf` and `NaN` injected at a few specific positions.
     * Purpose: Verify special value handling in a random data stream.
   * Row 18: Random Values with Injected Subnormals
     * Content: Mostly random numbers in the range `[-5.0, 5.0]`, with very small numbers (which convert to subnormals in bf16) injected at a few specific positions.
     * Purpose: Verify subnormal number handling in a random data stream.
   * Row 19: Random Large Positive Numbers
     * Content: Random numbers uniformly distributed in the range `[0.0, BF16_MAX/2]`.
     * Purpose: Verify accuracy and stability with large numerical values.
3. Activation Function Specific Pattern Tests (34 rows)
   This category of tests is designed for the mathematical properties and potential numerical challenges of each activation function.

   * **Softmax (6 rows)**

     * Row 20: Large Positive Increasing Sequence
       * Content: For example, a linear sequence starting from `100.0` with a step of `0.1`.
       * Purpose: Verify the stability of the `exp()` computation, especially the `max_k` normalization overflow prevention mechanism.
     * Row 21: Large Negative Decreasing Sequence
       * Content: For example, a linear sequence starting from -100.0 with a step of `-0.1`.
       * Purpose: Verify `exp()` underflowing to zero.
     * Row 22: Single Element Far Greater than Others
       * Content: A single element in the middle is `10.0`, the rest are `0.0`.
       * Purpose: Verify that the output is concentrated at the dominant element's position.
     * Row 23: Single `+Inf` Element
       * Content: A single element in the middle is `+Inf`, the rest are `0.0`.
       * Purpose: Verify that a single `+Inf` leads to an output of `1.0`.
     * Row 24: Two `+Inf` Elements
       * Content: Two elements are `+Inf`, the rest are `0.0`.
       * Purpose: Verify that multiple `+Inf` values lead to a uniform output distribution.
     * Row 25: Single `NaN` Element
       * Content: A single element in the middle is `NaN`, the rest are `0.0`.
       * Purpose: Verify `NaN` propagation.
   * **LayerNorm (6 rows)**

     * Row 26: Zero Mean, Non-Zero Variance
       * Content: For example, `[1.0, -1.0, 1.0, -1.0, ...]`.
       * Purpose: Verify mean and variance calculations.
     * Row 27: Values Close to Constant (Small Variance)
       * Content: For example, `[5.000, 5.001, 5.002, ...]`.
       * Purpose: Verify the role of `epsilon` in protecting against division by zero, and accuracy with small variance.
     * Row 28: Containing `Inf/NaN`
       * Content: Random numbers with injected `+Inf` and `NaN`.
       * Purpose: Verify the propagation of special values in mean and variance calculations.
     * Row 29: All Values Differ by Epsilon
       * Content: For example, `[0.0, EPSILON_LN, 2*EPSILON_LN, ...]`.
       * Purpose: Precisely test the `epsilon` boundary effect.
     * Row 30: Large Values
       * Content: Random numbers in the range `[0.0, BF16_MAX/10]`.
       * Purpose: Verify numerical stability with large values.
     * Row 31: Small Values (Non-subnormal)
       * Content: Random numbers in the range `[BF16_MIN_NORMAL, BF16_MIN_NORMAL*10]`.
       * Purpose: Verify numerical stability with small values.
   * **RMSNorm (6 rows)**

     * Row 32: Zero Mean, Non-Zero RMS
       * Content: For example, `[1.0, -1.0, 1.0, -1.0, ...]`.
       * Purpose: Verify RMS calculation.
     * Row 33: Values Close to Zero (Small RMS)
       * Content: For example, `[0.000, 0.001, 0.002, ...]`.
       * Purpose: Verify the role of `epsilon` in protecting against division by zero, and accuracy with small RMS.
     * Row 34: Containing `Inf/NaN`
       * Content: Random numbers with injected `+Inf` and `NaN`.
       * Purpose: Verify the propagation of special values in RMS calculation.
     * Row 35: All Values Differ by Epsilon (starting from zero)
       * Content: For example, `[0.0, EPSILON_RMS, 2*EPSILON_RMS, ...]`.
       * Purpose: Precisely test the `epsilon` boundary effect.
     * Row 36: Large Values
       * Content: Random numbers in the range `[0.0, BF16_MAX/10]`.
       * Purpose: Verify numerical stability with large values.
     * Row 37: Small Values (Non-subnormal)
       * Content: Random numbers in the range `[BF16_MIN_NORMAL, BF16_MIN_NORMAL*10]`.
       * Purpose: Verify numerical stability with small values.
   * **SiLU (4 rows)**

     * Row 38: Dense Sampling Near Zero
       * Content: For example, a linear sequence from `[-5.0, ..., 0.0, ..., 5.0]`.
       * Purpose: Verify the accuracy and behavior of the `sigmoid` approximation near zero.
     * Row 39: Large Positive Values
       * Content: For example, a linear sequence from `[10.0, ..., 100.0]`.
       * Purpose: Verify that SiLU(x) approaches x.
     * Row 40: Large Negative Values
       * Content: For example, a linear sequence from `[-10.0, ..., -100.0]`.
       * Purpose: Verify that SiLU(x) approaches 0.
     * Row 41: Containing `Inf/-Inf/NaN`
       * Content: Random numbers with injected `+Inf`, `-Inf`, `NaN`.
       * Purpose: Verify the handling of special values by the `sigmoid` and multiplication operations.
   * **GELU (4 rows)**

     * Row 42: Dense Sampling Near Zero
       * Content: For example, a linear sequence from `[-5.0, ..., 0.0, ..., 5.0]`.
       * Purpose: Verify the accuracy and behavior of the `erf` approximation near zero.
     * Row 43: Large Positive Values
       * Content: For example, a linear sequence from `[10.0, ..., 100.0]`.
       * Purpose: Verify that GELU(x) approaches x.
     * Row 44: Large Negative Values
       * Content: For example, a linear sequence from `[-10.0, ..., -100.0]`.
       * Purpose: Verify that GELU(x) approaches 0.
     * Row 45: Containing `Inf/-Inf/NaN`
       * Content: Random numbers with injected `+Inf`, `-Inf`, `NaN`.
       * Purpose: Verify the handling of special values by the `erf` approximation and multiplication operations.

    * **Elementwise Add/Multiply (8 rows - as operand X)**
        These rows serve as the first input X for Elementwise Add and Elementwise Multiply. The test script will generate the corresponding second input Y to form a complete test case.
        * Row 46: Random Positive Numbers
            * Content: Random numbers uniformly distributed in the range `[0.0, 10.0]`.
            * Purpose: General addition/multiplication test.
        * Row 47: Random Negative Numbers
            * Content: Random numbers uniformly distributed in the range `[-10.0, 0.0]`.
            * Purpose: General addition/multiplication test.
        * Row 48: Large Values
            * Content: Random numbers uniformly distributed in the range `[BF16_MAX/2, BF16_MAX]`.
            * Purpose: To be used with Y to test overflow.
        * Row 49: Small Values
            * Content: Random numbers uniformly distributed in the range `[BF16_MIN_NORMAL, BF16_MIN_NORMAL*10]`.
            * Purpose: To be used with Y to test underflow.
        * Row 50: Containing `+Inf`
            * Content: Random numbers with injected `+Inf`.
            * Purpose: To be used with Y to test `Inf + (-Inf)` or `0 * Inf`, etc.
        * Row 51: Containing `-Inf`
            * Content: Random numbers with injected `-Inf`.
            * Purpose: To be used with Y to test `Inf + (-Inf)`, etc.
        * Row 52: Containing `NaN`
            * Content: Random numbers with injected `NaN`.
            * Purpose: To be used with Y to test `NaN` propagation.
        * Row 53: Containing `0.0`
            * Content: Random numbers with injected `0.0`.
            * Purpose: To be used with Y to test `0 * Inf`, etc.

4. **Remaining Rows (10 rows)**
    * Rows 54-63: More Random or Complex Patterns
        * Content: Can be filled with random numbers from different distributions (e.g., log-normal distribution), different numerical ranges, or by repeating some existing test cases with different random seeds to increase test breadth. You can also design some "pathological" inputs, such as alternating extremely large and small values, to test for potential issues in pipelines and resource reuse.


#### Responsibilities of the Test Script

During the actual testing process, your Python test script will:

* Generate X_test_tensor
  Fill the `64 x 768` bf16 tensor according to the strategy above
* Generate Y_test_tensor (for Add/Mul)
  For Elementwise Add and Elementwise Multiply operations, generate a Y_test_tensor with a similar structure to X_test_tensor. Each row of Y_test_tensor should be paired with the corresponding row in X_test_tensor to form a meaningful test pair (e.g., X is a large positive number, Y is a large positive number to test for overflow; X is `+Inf`, Y is `-Inf` to test for a `NaN` result).
* Generate Golden Reference Output
  Use PyTorch's bfloat16 data type to apply the corresponding activation function to each row of X_test_tensor (and the corresponding row of Y_test_tensor) to compute the golden reference output Y_ref.
* Data Transfer
  Pack the bf16 bit patterns of X_test_tensor (and the relevant rows of Y_test_tensor) into the format required for the AXI Memory-Mapped interface and transfer them to the FPGA.
* Read FPGA Results
  Read the computation results Y_fpga from the FPGA.
* Compare Results
  Convert Y_fpga to bfloat16 floating-point numbers and compare them with Y_ref, compute the relative L2 error $\epsilon_f$, and evaluate the accuracy score $A_f$ according to the scoring criteria. At the same time, perform an element-by-element precise comparison for special values.
