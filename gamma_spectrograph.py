import numpy as np
import sys
import pandas as pd
import ast

def r2_rest(x_data, y_data, y_predicted):

    """
    INPUTS:
    x_data = x-axis original data (Energy channels)
    y_data = y-axis original data (Counts)
    y_predicted = predicted data of any fit (counts)

    OUTPUTS:
    R2 = R^2 error value
    
    """
    # Convert arrays to numpy arrays for element-wise calculations.
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    y_predicted = np.array(y_predicted)
    
    mean_data = np.mean(y_data)
    ssTot = sum((y_data - mean_data)**2)
    ssRes = sum((y_data - y_predicted)**2)

    R2 = 1 - (ssRes / ssTot)
    return R2

def channelToEnergy(x):
    """
    INPUTS:
    x = the channels
    OUTPUTS:
    corrected_x = Corrected channels, energy levels.
    """
    x = np.array(x)
    corrected_x = x
    return corrected_x

def efficiencyCorrection(y):
    """Efficiency correction, if required.
    
    """
    energy = channelToEnergy(y)                                                 # Conversion from channels to energy

    eff = 1
    # Apply the correction to the original array
    corrected_y = y/eff
    return corrected_y

def levenberg_marquardt_gauss_fit(x_data, y_data, initial_params, max_iter=100, tol=1e-6):

    def gaussian(x, amplitude, mean, std):
        """Gaussian function."""
        return amplitude * np.exp(-(x - mean) ** 2 / (2 * std ** 2))

    def jacobian(x, amplitude, mean, std):
        """Calculate the Jacobian matrix."""
        J = np.zeros((len(x), 3))
        J[:, 0] = np.exp(-(x - mean) ** 2 / (2 * std ** 2))  # d(f)/d(amplitude)
        J[:, 1] = amplitude * (x - mean) / std ** 2 * np.exp(-(x - mean) ** 2 / (2 * std ** 2))  # d(f)/d(mean)
        J[:, 2] = amplitude * (x - mean) ** 2 / std ** 3 * np.exp(-(x - mean) ** 2 / (2 * std ** 2))  # d(f)/d(std)
        return J

    amplitude, mean, std = initial_params
    params = np.array([amplitude, mean, std])
    n_params = len(params)
    lambd = 1.0  # Initial damping factor
    v = 2.0  # Update factor

    for _ in range(max_iter):
        # Predicted values and residuals
        y_pred = gaussian(x_data, *params)
        residuals = y_data - y_pred

        # Jacobian matrix
        J = jacobian(x_data, *params)

        # Gradient and Hessian approximation
        A = J.T @ J
        g = J.T @ residuals

        # Check for convergence
        if np.linalg.norm(g, ord=np.inf) < tol:
            break

        # Adjust parameters using LM update rule
        I = np.eye(n_params)
        try:
            # Solve system instead of inverting
            delta = np.linalg.solve(A + lambd * I, g)
        except np.linalg.LinAlgError:
            continue

        # Check if the new parameters improve the solution
        new_params = params + delta
        new_y_pred = gaussian(x_data, *new_params)
        new_residuals = y_data - new_y_pred
        rho = (np.linalg.norm(residuals) ** 2 - np.linalg.norm(new_residuals) ** 2) / (delta @ (lambd * delta + g))

        if rho > 0:
            # Update the parameters
            params = new_params
            lambd = max(lambd * max(1/3, 1 - (2*rho - 1)**3), 1e-10)  # Prevent lambd from becoming too small
            v = 2
        else:
            lambd = min(lambd * v, 1e10)  # Cap lambd at a reasonable large value
            v *= 2

    gauss_fit = gaussian(x_data, params[0], params[1], params[2])

    return gauss_fit, params[0], params[1], params[2]

def average(A, m):
    #"""Smooths input function by averaging neighboring values."""

    # define empty array same length as data
    B = np.zeros(len(A))

    # loop over data points and sum all points m either side of current one
    for i in range(len(A)):
        for j in range(i - m, i + m):
            try:
                B[i] += A[j]
            # if tries to go outside of boundaries of data, do nothing
            except IndexError:
                pass

    return B

def Peak_Finder(x, y, confidence = 5, FWHM = 15, intensity_threshold = 0.001, z = 5):
    def second_diff(N, m, z):
        #"""Finds second difference of spectrum, discrete analogue to second derivative."""
        # empty list with same size as data
        N = np.asarray(N)
    
        S = np.zeros(len(N))   
        # loop over data points and for each take the second difference
        for i in range(len(N)):
            try:
                S[i] = N[i + 1] - 2 * N[i] + N[i - 1]
            # if at the ends of the array need to do different calculation to avoid error
            except IndexError:
                if i == 0:
                    S[i] = N[i + 1] - 2 * N[i]
                else:
                    S[i] = N[i - 1] - 2 * N[i]
    
        # smooth the second difference using Mariscotti's values
        for i in range(z):
            S = average(S, m)
    
        return S
    
    def standard_dev_S(N, m, z):
        """Finds standard deviation of second difference."""
        # convert to NumPy array to allow negative indexing
        N = np.asarray(N)
        
        F = np.zeros(len(N))
    
        # loop over data points and take variance-like quantity
        for i in range(len(N)):
            try:
                F[i] = N[i + 1] + 4 * N[i] + N[i - 1]
            except IndexError:
                if i == 0:
                    F[i] = N[i + 1] + 4 * N[i]
                else:
                    F[i] = N[i - 1] + 4 * N[i]
    
        # smoothing
        for _ in range(z):
            F = average(F, m)
    
        # return square root of F as std dev
        return np.sqrt(F)
    
    w = int(0.6 * FWHM)
    if (w % 2) == 0:
        w += 1
    m = int((w - 1) / 2)
    S = second_diff(N = y, m = m, z = z)
    std_S = standard_dev_S(N = y, m = m, z = z)
    
    # If the second difference is negative (indicates gaussian-like feature) and magnitude is greater than \ \
    # the standard deviation multiplied by the confidence factor then can say a peak has been found, \ \
    # if peak is greater than a % of spectrum's intensity, set index of peak centre to 1 in the signals array \ \
    idx_peak = []
    signals = np.zeros_like(y)
    for idx, val in enumerate(y):
        if abs(S[idx]) > std_S[idx] * confidence and S[idx]<0:
            try:
                if y[idx] == max(y[idx - FWHM : idx + FWHM]) and y[idx] >= (intensity_threshold / 100) * max(y):
                    signals[idx] = 1
                    idx_peak.append(idx)
            except ValueError:
                pass
            
    return idx_peak    

def Simpsons_Integration(x, y):
    # Calculate the number of segments
    n = len(x) - 1
    if n%2!=0:
        n+=1
    
    h = (x[-1] - x[0]) / n  # step size

    # Apply Simpson's rule 
    integral = y[0] + y[-1]

    for i in range(1, n, 2):
        integral += 4 * y[i]
    for i in range(2, n-1, 2):
        integral += 2 * y[i]

    integral *= h / 3

    return integral

def linear_regression_maximum_likelihood(x,y):
    n = len(x)
    x = np.array(x)
    y = np.array(y)
    a_numerator = np.sum(y) * np.sum(x**2) - np.sum(y * x) * np.sum(x)
    a_denominator = n * np.sum(x**2) - (np.sum(x))**2
    a = a_numerator / a_denominator
    
    b_numerator = n * np.sum(x * y) - np.sum(y) * np.sum(x)
    b_denominator = n * np.sum(x**2) - (np.sum(x))**2
    b = b_numerator / b_denominator
    
    return a, b

def Linear_Gauss_at_Peaks(x, y, window_size = 140, plot = True, confidence = 7, FWHM = 45, intensity_threshold = 0.001, z = 5):
    
    # Step 1: Find the peaks
    peak_index_list = Peak_Finder(x,y, confidence = confidence, FWHM = FWHM, intensity_threshold = intensity_threshold, z = z)
    
    def Near_Peak_Data(peak_idx, x, y, window_size):
        # get the data around the peak
        x_array = x[peak_idx - window_size : peak_idx + window_size] 
        y_array = y[peak_idx - window_size : peak_idx + window_size] 
        
        return (x_array, y_array)

    # To contain the x-array near the peaks.
    x_peak_list = []

    # Lists for gaussian fit
    fitted_gauss_list = []
    integral_gauss_list = []
    mean_list = []
    std_list = []

    # Lists for the linear fit
    fitted_linear_list = []
    integral_linear_fit_list = []
    
    y_peak_list = []
    combined_fit_list = []
    R2_combined_list = []

    # For each peak, calculate the parameters, append them to the corresponding lists.
    for idx, peak_index in enumerate(peak_index_list):

        # Determine the index of the peak_index_list.
        x_peak, y_peak = Near_Peak_Data(peak_index, x, y, window_size)
        y_peak_list.append(y_peak)

        ##########################
        #     Linear fit Part    #
        ##########################
        linear_window = 3  # Gets the first and last indices, adjust this based on the size of your data
        x_linear_data = np.concatenate((x_peak[:linear_window], x_peak[-linear_window:]))  # Get first and last indices to rule out the peak data
        y_linear_data = np.concatenate((y_peak[:linear_window], y_peak[-linear_window:]))

        # Perform linear regression
        a, b = linear_regression_maximum_likelihood(x_linear_data, y_linear_data)  # y = b * x + a

        fitted_linear = b * np.array(x_peak) + a

        # Compute the integral using Simpsons_Integration
        integral_linear_fit = 1#Simpsons_Integration(x_peak, fitted_linear)

        # Calculate RMSE for the linear fit
        y_linear_fit_data = b * np.array(x_linear_data) + a       # Get the linear fit interpolation at the fitting points
        #rmse_linearfit = np.sqrt(np.mean((y_linear_fit_data - y_linear_data) ** 2))

        integral_linear_fit_list.append(integral_linear_fit)
        #rmse_linearfit_list.append(rmse_linearfit)
        fitted_linear_list.append(y_linear_fit_data)
        
        ##########################
        #     Gauss fit Part     # 
        ##########################
        # In order to do the first guess right, use the statistical data of the energy and counts.
        peak_ampl = max(y_peak)
        mean_x = np.mean(x_peak)
        std_x =  2.35 * mean_x#* FWHM 
        initial_params = [peak_ampl, mean_x, std_x]  # Example initial guesses
    
        fitted_gauss, amplitude, mean, std = levenberg_marquardt_gauss_fit(x_peak, y_peak - fitted_linear, initial_params)
        integral_gauss_fit = 1#Simpsons_Integration(x_peak, fitted_gauss)
        
        combined_fit = fitted_gauss + fitted_linear                                   # The function described in equation (1) Mariscotti (1967)
        R2_combined = r2_rest(x_peak, combined_fit, y_peak)
        
        # Append the values into the lists to be returned as a result.
        x_peak_list.append(x_peak)
        fitted_gauss_list.append(fitted_gauss)
        integral_gauss_list.append(integral_gauss_fit)
        mean_list.append(mean)
        std_list.append(std)

        combined_fit_list.append(combined_fit)
        R2_combined_list.append(R2_combined)

    return x_peak_list, y_peak_list, peak_index_list, fitted_gauss_list, integral_gauss_list, mean_list, std_list, fitted_linear_list, combined_fit_list, R2_combined_list

def get_matching_isotope_parameters_by_energy(energy=511.0, tolerance=10):
    """
    Retrieves isotope data based on a given energy value and tolerance range.
    It searches for isotopes matching the specified energy range and returns their properties.
    """
    # Load the CSV
    df = pd.read_csv("Filtered_IsotopeLibrary_Grouped.csv")  

    # Convert string representations of lists into actual lists
    df["Energy"] = df["Energy"].apply(ast.literal_eval)

    # Explode the lists so that each energy value gets its own row
    df = df.explode("Energy")

    # Convert energy column to float
    df["Energy"] = df["Energy"].astype(float)

    # Filter by energy range
    df_energy = df[(df["Energy"] >= energy - tolerance) & 
                   (df["Energy"] <= energy + tolerance)]

    # Get unique isotopes in the filtered data
    isotopes_by_energy = df_energy["Isotope"].unique()

    # Build a dictionary with isotopes and all their energy values
    isotope_dict = {iso: df[df["Isotope"] == iso]["Energy"].values for iso in isotopes_by_energy}

    # Convert to DataFrame
    isotope_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in isotope_dict.items()]))

    # Fill NaN values with 0 to match the expected format
    isotope_df = isotope_df.fillna(0)

    # Ensure isotope_peaks_all is a list of NumPy arrays
    isotope_peaks_all = [np.array(isotope_df[col].values) for col in isotope_df.columns]

    # Get isolated energy peaks
    isotope_peaks_isolated = df_energy["Energy"].values

    # Get isotope names
    isotope_names = isotope_df.columns.values

    return isotope_names, isotope_peaks_all, isotope_peaks_isolated, isotope_df


def get_matched_peaks_data(x, y, tolerance = 50, window_size=200, confidence=5, FWHM=45, intensity_threshold=0.001, z=5, r2_threshold = 0.6):

    """
    INPUTS:
    x = Energy channels on x-axis
    y = c
    Produces Isotope name, peak energy, maximum count list (y_peak_max_list is for testing purposes, may be excluded afterwards).
    
    """
    #isotope_name = df_file[5:-10]
    
    # Perform Gaussian fitting and peak detection
    x_peak_list, y_peak_list, peak_index_list, fitted_gauss_list, integral_gauss_list, mean_list, std_list, fitted_linear_list, combined_fit_list, R2_combined_list  = Linear_Gauss_at_Peaks(x, y, window_size=window_size, plot=False, confidence=confidence, FWHM=FWHM, intensity_threshold=intensity_threshold, z=z)
    
    matching_isotope_names = []
    matching_isotope_peaks = []
    matching_isotope_R2 = []

    detected_parameter_dict = {}
    detected_peak_list = []
    detected_integral_list = []
    detected_FWHM_list = []
    
    y_peak_max_list = []
    matching_idx_list = []
    
    for idx, peak_val in enumerate(mean_list):
        isotope_names_library, isotope_peaks_library, isotope_peaks_isolated, isotope_df = get_matching_isotope_parameters_by_energy(energy=peak_val, tolerance=tolerance)
        if ( isotope_names_library.any() ) and ( R2_combined_list[idx] > r2_threshold):
            matching_isotope_names.append(isotope_names_library)
            matching_isotope_peaks.append(isotope_peaks_isolated)
            matching_isotope_R2.append(R2_combined_list[idx])

            detected_peak_list.append(mean_list[idx])
            detected_integral_list.append(integral_gauss_list[idx])
            detected_FWHM_list.append(abs(std_list[idx] / 2.35) )

            y_peak_max_list.append(max(y_peak_list[idx]))
            matching_idx_list.append(peak_index_list[idx])
    
    matching_isotope_peaks = [
        [arr[arr != 0] for arr in sublist]
        for sublist in matching_isotope_peaks
    ]
    
    detected_parameter_dict["Isotope"] = matching_isotope_names
    detected_parameter_dict["Peak"] = detected_peak_list
    detected_parameter_dict["R2"] = matching_isotope_R2
    detected_parameter_dict["FWHM"] = detected_FWHM_list

    # Initialize an empty dictionary
    grouped_isotopes = {}
    
    # Populate the dictionary
    for isotopes, peak in zip(detected_parameter_dict['Isotope'], detected_parameter_dict['Peak']):
        for isotope in isotopes:
            if isotope not in grouped_isotopes:
                grouped_isotopes[isotope] = []
            grouped_isotopes[isotope].append(peak)
    
    isotope_lib_grouped = pd.read_csv("Filtered_IsotopeLibrary_Grouped.csv")
    isotope_lib_grouped["Energy"] = isotope_lib_grouped["Energy"].apply(ast.literal_eval)
    
    real_matched_isotope_dict = {}
    # Filter grouped_isotopes to match library peaks more closely
    for matched_isotope_name in grouped_isotopes.keys():
        library_peaks = isotope_lib_grouped["Energy"][
            isotope_lib_grouped["Isotope"] == matched_isotope_name
        ].values[0]
    
        # Sort detected peaks by proximity to library peaks
        grouped_isotopes[matched_isotope_name].sort(
            key=lambda x: min(abs(x - lib_peak) for lib_peak in library_peaks)
        )
    
        # Retain only the closest `len_isotope_lib_grouped` peaks
        grouped_isotopes[matched_isotope_name] = grouped_isotopes[matched_isotope_name][
            :len(library_peaks)
        ]

    for matched_isotope_name in grouped_isotopes.keys():
        len_grouped_isotopes = len(grouped_isotopes[matched_isotope_name])
        len_isotope_lib_grouped = len(isotope_lib_grouped["Energy"][isotope_lib_grouped["Isotope"] == matched_isotope_name].values[0])
    
        if len_grouped_isotopes == len_isotope_lib_grouped == 1:
            real_matched_isotope_dict[matched_isotope_name] = grouped_isotopes[matched_isotope_name]
        elif len_grouped_isotopes == len_isotope_lib_grouped == 2:
            real_matched_isotope_dict[matched_isotope_name] = grouped_isotopes[matched_isotope_name]
        elif len_isotope_lib_grouped > 2 and (len_grouped_isotopes >= int(len_isotope_lib_grouped / 2)):
            real_matched_isotope_dict[matched_isotope_name] = grouped_isotopes[matched_isotope_name]
    
    # Filter detected isotopes
    filtered_isotopes = []
    filtered_peaks = []
    filtered_FWHM = []
    filtered_y_peak_max = []
    filtered_r2 = []

    for idx_1, isotopes in enumerate(detected_parameter_dict["Isotope"]):
        current_isotopes = []
        current_peaks = []
        current_y_peak_max = []
        current_FWHM = []
        current_r2 = []
        for idx_2, isotope in enumerate(isotopes):
            if isotope in real_matched_isotope_dict:
                current_isotopes.append(detected_parameter_dict["Isotope"][idx_1][detected_parameter_dict["Isotope"][idx_1] == isotope][0])
                current_peaks.append(detected_parameter_dict["Peak"][idx_1])
                current_FWHM.append(detected_parameter_dict["FWHM"][idx_1])
                current_y_peak_max.append(y_peak_max_list[idx_1])
                current_r2.append(matching_isotope_R2[idx_1])
                
        if current_peaks:
            filtered_isotopes.append(current_isotopes)
            filtered_peaks.append(current_peaks)
            filtered_FWHM.append(current_FWHM)

            filtered_y_peak_max.append(current_y_peak_max)
            filtered_r2.append(current_r2)

    detected_parameter_dict["Isotope"] = filtered_isotopes
    detected_parameter_dict["Peak"] = filtered_peaks
    detected_parameter_dict["R2"] = filtered_r2
    detected_parameter_dict["FWHM"] = filtered_FWHM

    return detected_parameter_dict, filtered_y_peak_max, x, y, matching_idx_list 