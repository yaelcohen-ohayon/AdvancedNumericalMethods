# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.integrate as integrate

# def calc_ddf(lamda, eta, dtheta, theta):
#     """Calculate the second derivative of f (equation 8 from the paper).
    
#     Args:
#         lamda (float): Lambda parameter.
#         eta (float): Similarity variable.
#         dtheta (float): First derivative of theta.
#         theta (float): Temperature function.
    
#     Returns:
#         float: Second derivative of f (f'').
#     """
#     ddf = -((lamda - 2)/3 * eta * dtheta + lamda * theta)
#     return ddf

# def calc_ddtheta(lamda, df, theta, dtheta, f):
#     """Calculate the second derivative of theta (equation 9 from the paper).
    
#     Args:
#         lamda (float): Lambda parameter.
#         df (float): First derivative of f.
#         theta (float): Temperature function.
#         dtheta (float): First derivative of theta.
#         f (float): Stream function.
    
#     Returns:
#         float: Second derivative of theta (theta'').
#     """
#     ddtheta = lamda * df * theta - (lamda + 1)/3 * f * dtheta
#     return ddtheta

# def derivatives(eta, y):
#     """Compute the system of first-order ODEs for the shooting method.
    
#     Args:
#         eta (float): Independent variable (similarity variable).
#         y (np.ndarray): State vector [f, f', theta, theta'].
    
#     Returns:
#         np.ndarray: Derivatives [f', f'', theta', theta''].
#     """
#     # y = [f, f', theta, theta']
#     f = y[0]
#     df = y[1]
#     theta = y[2]
#     dtheta = y[3]
    
#     # משוואה 8 מהמאמר
#     ddf = calc_ddf(LAMBDA, eta, dtheta, theta)
    
#     # משוואה 9 מהמאמר
#     ddtheta = calc_ddtheta(LAMBDA, df, theta, dtheta, f)
    
#     return np.array([df, ddf, dtheta, ddtheta])

# def RK4_solver(fun, eta, y, h):
#     """Fourth-order Runge-Kutta solver for ODE systems.
    
#     Args:
#         fun (callable): Function that computes derivatives dy/deta = fun(eta, y).
#         eta (float): Current value of independent variable.
#         y (np.ndarray): Current state vector.
#         h (float): Step size.
    
#     Returns:
#         np.ndarray: Updated state vector at eta + h.
#     """
#     k1 = fun(eta, y)
#     k2 = fun(eta + 0.5*h, y + 0.5*h*k1)
#     k3 = fun(eta + 0.5*h, y + 0.5*h*k2)
#     k4 = fun(eta + h, y + h*k3)

#     y_new = y + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
#     return y_new

# def integrate_system(fw, df_guess, dtheta_guess):
#     """Integrate the ODE system using RK4 method.
    
#     Args:
#         fw (float): Initial value of f at eta=0.
#         df_guess (float): Guessed value of f' at eta=0.
#         dtheta_guess (float): Guessed value of theta' at eta=0.
    
#     Returns:
#         tuple: (eta_array, results_array) where:
#             - eta_array: Array of eta values.
#             - results_array: Array of shape (N, 4) with [f, f', theta, theta'] at each eta.
#     """
#     y0 = np.array([fw, df_guess, 1.0, dtheta_guess]) # תנאי התחלה: f(0)=fw, theta(0)=1
#     eta = 0.0
#     results = [y0.copy()] # לאחסון התוצאות
#     etas = [eta] # לאחסון ערכי eta

#     for i in range(STEPS):
#         y0 = RK4_solver(derivatives, eta, y0, H)
#         eta += H
#         results.append(y0.copy())
#         etas.append(eta)
#         # if y0[2] < 1e-8:  # עצירה מוקדמת אם theta קטן מאוד
#         #     return np.array(etas), np.array(results)
        
#     return np.array(etas), np.array(results)

# def newton_raphson_solver(x1, x2Df, x2Theta, y1, y2Df, y2Theta, delta, initGuess, errors):
#     """Newton-Raphson solver for updating initial guess in shooting method.
    
#     Args:
#         x1 (float): Baseline f' at boundary.
#         x2Df (float): f' at boundary with perturbed df_guess.
#         x2Theta (float): theta at boundary with perturbed df_guess.
#         y1 (float): Baseline theta at boundary.
#         y2Df (float): f' at boundary with perturbed dtheta_guess.
#         y2Theta (float): theta at boundary with perturbed dtheta_guess.
#         delta (float): Perturbation size.
#         initGuess (np.ndarray): Current guess [df_guess, dtheta_guess].
#         errors (np.ndarray): Error vector [f'_error, theta_error].
    
#     Returns:
#         np.ndarray: Updated guess [df_guess, dtheta_guess].
#     """
#     J = np.zeros((2, 2)) # הגדרת יעקוביאן לניוטון רפסון
#     # J[i,j] = ∂(error_i)/∂(guess_j)
#     # error vector = [f'_error, theta_error]
#     # guess vector = [df_guess, dtheta_guess]
#     J[0, 0] = (x2Df - x1) / delta      # ∂(f'_error)/∂(df_guess)
#     J[0, 1] = (y2Df - x1) / delta      # ∂(f'_error)/∂(dtheta_guess)
#     J[1, 0] = (x2Theta - y1) / delta   # ∂(theta_error)/∂(df_guess)
#     J[1, 1] = (y2Theta - y1) / delta   # ∂(theta_error)/∂(dtheta_guess)
    
#     try:
#         update = np.linalg.solve(J, errors)
#         guess = initGuess - update
#         return guess
#     except np.linalg.LinAlgError:
#         print("Singular Matrix encountered. Try different initial guesses.")
#         return initGuess

# def shooting_method(lambdaArray, fw, oneOverC, initGuessArray, TARGET_THETA, EPS=1e-3):
#     """Solve boundary value problem using shooting method with Newton-Raphson iteration.
    
#     Args:
#         lambdaArray (np.ndarray): Array of lambda values to solve for.
#         fw (np.ndarray): Array of wall velocity values.
#         oneOverC (np.ndarray): Array of 1/C values (suction/injection parameter).
#         initGuessArray (np.ndarray): Initial guess table with shape (N, 6).
#         TARGET_THETA (float): Target value for theta at infinity (typically 0).
#         EPS (float, optional): Convergence tolerance. Defaults to 1e-3.
    
#     Returns:
#         tuple: (resArray, paramsArray) where:
#             - resArray: List of tuples (eta_array, solution_array) for each case.
#             - paramsArray: List of parameter dictionaries for each case.
#     """
#     global ETA_INF, STEPS, H, LAMBDA
#     resArray = []
#     paramsArray = []  # Store parameters for each run
#     # Full arrays matching the data table structure
#     full_lambda = [0.5, 2.0]
#     full_fw = [-1.0, 0.0, 1.0]
#     full_oneOverC = [1.0, 2.0, 5.0, 8.0]
    
#     for lIdx, lamda in enumerate(lambdaArray):
#         for fwIdx, fw_val in enumerate(fw):
#             for cIdx, c in enumerate(oneOverC):
#                 if lamda == 2.0 and fw_val == -1.0 and (c == 5.0 or c == 8.0):
#                     print(f"Skipping known divergent case: Lambda={lamda}, fw={fw_val}, C={c}")
#                     # continue
#                 # Calculate correct index into data table
#                 lambda_idx_full = full_lambda.index(lamda)
#                 fw_idx_full = full_fw.index(fw_val)
#                 c_idx_full = full_oneOverC.index(c)
#                 ii = lambda_idx_full * 12 + fw_idx_full * 4 + c_idx_full
                
#                 # ETA_INF = initGuessArray[ii, 5] * 1.1
#                 ETA_INF = 10
#                 H = 0.1
#                 STEPS = int(ETA_INF / H)
#                 LAMBDA = lamda
#                 C_PARAM_FINAL = c**(-2/3)
#                 C_PARAM = c**(1/3)
#                 initDThetaGuess = initGuessArray[ii, 3]
#                 initDfGuess = initGuessArray[ii, 4]
#                 # Boundary condition: f(0) = fw (no scaling needed for shooting method)
#                 fwToUse = fw_val
#                 res = integrate_system(fwToUse, initDfGuess, initDThetaGuess)
#                 resArray.append(res)
#                 paramsArray.append({'lambda': lamda, 'fw': fw_val, 'oneOverC': c})

#                 boundryValues = res[1][-1, :]
#                 fFinal = boundryValues[0]
#                 dfFinal = boundryValues[1]
#                 thetaFinal = boundryValues[2]
#                 dthetaFinal = boundryValues[3]

#                 fTagError = dfFinal - C_PARAM_FINAL
#                 thetaError = thetaFinal - TARGET_THETA

#                 errNorm = np.sqrt(fTagError**2 + thetaError**2)
                
#                 if errNorm <= EPS:
#                     print(f"Converged for Lambda={lamda}, fw={fw_val}, C={c}. Error Norm={errNorm}")
#                     continue  # Skip to next iteration, already converged
                
#                 # Initialize divergence check - will be updated in first loop iteration
#                 divergence_detected = False
                
#                 while errNorm > EPS and not divergence_detected:
#                     delta = 1e-7
#                     resDf = integrate_system(fwToUse, initDfGuess + delta, initDThetaGuess)
#                     resDtheta = integrate_system(fwToUse, initDfGuess, initDThetaGuess + delta)

#                     boundryDf_Df = resDf[1][-1, 1]
#                     boundryDf_Theta = resDf[1][-1, 2]
#                     boundryDtheta_Df = resDtheta[1][-1, 1]
#                     boundryDtheta_Theta = resDtheta[1][-1, 2]

#                     valArr = np.array([boundryDf_Df, boundryDf_Theta, boundryDtheta_Df, boundryDtheta_Theta])
                    
#                     # Check for divergence
#                     if (np.any(np.abs(valArr) > 10)):
#                         print(f"Divergence detected for Lambda={lamda}, fw={fw_val}, C={c}")
#                         divergence_detected = True
#                         break

#                     new_res = newton_raphson_solver(dfFinal, boundryDf_Df, boundryDf_Theta, thetaFinal, boundryDtheta_Df, boundryDtheta_Theta, delta, np.array([initDfGuess, initDThetaGuess]), np.array([fTagError, thetaError]))
                    
#                     initDfGuess, initDThetaGuess = new_res
#                     res = integrate_system(fwToUse, initDfGuess, initDThetaGuess)
#                     resArray[-1] = res

#                     boundryValues = res[1][-1, :]
#                     fFinal = boundryValues[0]
#                     dfFinal = boundryValues[1]
#                     thetaFinal = boundryValues[2]
#                     dthetaFinal = boundryValues[3]

#                     fTagError = dfFinal - C_PARAM_FINAL
#                     thetaError = thetaFinal - TARGET_THETA

#                     errNorm = np.sqrt(fTagError**2 + thetaError**2)
#                     if errNorm <= EPS:
#                         print(f"Converged for Lambda={lamda}, fw={fw_val}, C={c}. Final Error Norm={errNorm}")
#                     # print(f"Iterating for Lambda={lamda}, fw={fw_val}, 1/C={c}, Error Norm={errNorm}")

#                 # plt.figure()
#                 # plt.plot(res[0], res[1][:, :])  # Plotting eta vs f
#                 # plt.grid()
#                 # plt.xlabel('Eta')
#                 # plt.ylabel('f(eta)')
#                 # plt.title(f'Lambda={lamda}, C^1/3={C_PARAM_FINAL}, fw={fw_val}')
#                 # plt.legend(['f', "f'", 'theta', "theta'"])

#     return resArray, paramsArray

# def finite_difference_method(lambdaArray, fw, oneOverC, initGuessArray, TARGET_THETA, EPS=1e-5):
#     """Solve boundary value problem using finite difference method with iterative solving.
    
#     Uses a block iteration scheme where f is integrated from equation (8) and theta
#     is solved using Gauss-Seidel iteration from equation (9).
    
#     Args:
#         lambdaArray (np.ndarray): Array of lambda values to solve for.
#         fw (np.ndarray): Array of wall velocity values.
#         oneOverC (np.ndarray): Array of 1/C values (suction/injection parameter).
#         initGuessArray (np.ndarray): Initial guess table with shape (N, 6).
#         TARGET_THETA (float): Target value for theta at infinity (typically 0).
#         EPS (float, optional): Convergence tolerance. Defaults to 1e-5.
    
#     Returns:
#         tuple: (resArray, paramsArray) where:
#             - resArray: List of tuples (eta_array, solution_array) for each case.
#             - paramsArray: List of parameter dictionaries for each case.
#     """
#     global ETA_INF, STEPS, H, LAMBDA, MAX_ITER
#     MAX_ITER = 30000
#     resArray = []
#     paramsArray = []  # Store parameters for each run
#     # Full arrays matching the data table structure
#     full_lambda = [0.5, 2.0]
#     full_fw = [-1.0, 0.0, 1.0]
#     full_oneOverC = [1.0, 2.0, 5.0, 8.0]

#     def cumtrapz(y, h):
#         # cumulative trapezoid integral from 0..i
#         out = np.zeros_like(y)
#         out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * h)
#         return out

#     def d1_center(y, h):
#         dy = np.zeros_like(y)
#         dy[1:-1] = (y[2:] - y[:-2]) / (2.0 * h)
#         # 2nd-order one-sided at boundaries
#         dy[0]  = (-3*y[0] + 4*y[1] - y[2]) / (2.0*h)
#         dy[-1] = ( 3*y[-1] - 4*y[-2] + y[-3]) / (2.0*h)
#         return dy

#     for lamda in lambdaArray:
#         for fw_val in fw:
#             for c in oneOverC:
#                 # Calculate correct index into data table
#                 lambda_idx_full = full_lambda.index(lamda)
#                 fw_idx_full = full_fw.index(fw_val)
#                 c_idx_full = full_oneOverC.index(c)
#                 ii = lambda_idx_full * 12 + fw_idx_full * 4 + c_idx_full
                
#                 # ETA_INF = initGuessArray[ii, 5] * 2
#                 ETA_INF = 7.0
#                 # ---- FIX: consistent grid ----
#                 H = 0.1
#                 N = int(np.ceil(ETA_INF / H))
#                 if N < 6:
#                     N = 6
#                     ETA_INF = N * H
#                 eta = np.linspace(0.0, ETA_INF, N + 1)
#                 H = eta[1] - eta[0]
#                 STEPS = N + 1
#                 LAMBDA = lamda

#                 # oneOverC = 1/C
#                 C_PARAM_FINAL = c**(-2/3)     # = C^(2/3) : f'(inf)
#                 C_PARAM = c**(1/3)            # = C^(-1/3)
#                 fwToUse = fw_val * C_PARAM    # f(0)

#                 # init guesses
#                 theta = np.exp(-eta)
#                 theta[0] = 1.0
#                 theta[-1] = 0.0

#                 # f init consistent with far-field slope
#                 f = fwToUse + C_PARAM_FINAL * eta
#                 fp = np.full_like(eta, C_PARAM_FINAL)

#                 omega_theta = 0.8  # relaxation inside theta GS sweep
#                 omega_block = 1.0  # optional block relaxation (theta <- mix(theta_old, theta_new))
#                 inv_h2 = 1.0 / (H * H)
#                 inv_2h = 1.0 / (2.0 * H)

#                 converged = False
#                 for k in range(MAX_ITER):
#                     theta_old = theta.copy()
#                     f_old = f.copy()

#                     # ===== Block 1: integrate f from eq (8) using current theta =====
#                     dtheta = d1_center(theta, H)

#                     # eq (8): f'' = -((λ-2)/3 * η * θ' + λ θ)
#                     fpp = -(((LAMBDA - 2.0) / 3.0) * eta * dtheta + LAMBDA * theta)

#                     # enforce far-field slope: fp(0) = fp_inf - ∫_0^∞ f'' dη
#                     I = np.trapz(fpp, x=eta)
#                     fp0 = C_PARAM_FINAL - I

#                     fp = fp0 + cumtrapz(fpp, H)
#                     f = fwToUse + cumtrapz(fp, H)

#                     # ===== Block 2: GS sweep for theta from eq (9) using updated f, fp =====
#                     theta[0] = 1.0
#                     theta[-1] = 0.0

#                     for i in range(1, STEPS - 1):
#                         # eq (9): theta'' + p theta' + q theta = 0
#                         p = ((LAMBDA + 1.0) / 3.0) * f[i]
#                         q = -LAMBDA * fp[i]

#                         coeff_minus = inv_h2 - p * inv_2h
#                         coeff_plus  = inv_h2 + p * inv_2h
#                         denom = -2.0 * inv_h2 + q

#                         # safety
#                         if abs(denom) < 1e-300:
#                             denom = np.copysign(1e-300, denom)

#                         val = -(coeff_minus * theta[i - 1] + coeff_plus * theta[i + 1]) / denom
#                         theta[i] = (1.0 - omega_theta) * theta[i] + omega_theta * val

#                     theta[0] = 1.0
#                     theta[-1] = 0.0

#                     # optional block relaxation on theta (helps if oscillatory)
#                     if omega_block < 1.0:
#                         theta = (1.0 - omega_block) * theta_old + omega_block * theta
#                         theta[0] = 1.0
#                         theta[-1] = 0.0

#                     err_theta = np.max(np.abs(theta - theta_old))
#                     err_f = np.max(np.abs(f - f_old))

#                     if k % 1000 == 0:
#                         fp_inf_est = (3*f[-1] - 4*f[-2] + f[-3]) / (2.0 * H)
#                         print(f"Iter {k}: Theta Error={err_theta:.2e}, f'(inf)={fp_inf_est:.6f}, STEPS={STEPS}")

#                     if max(err_theta, err_f) < EPS:
#                         print(f"  Converged in {k} iterations.")
#                         dtheta_out = d1_center(theta, H)
#                         resArray.append([eta, np.asarray([f, fp, theta, dtheta_out]).T])
#                         paramsArray.append({'lambda': lamda, 'fw': fw_val, 'oneOverC': c})
#                         converged = True
#                         break

#                 if not converged:
#                     print("Did not converge.")
#                     dtheta_out = d1_center(theta, H)
#                     resArray.append([eta, np.asarray([f, fp, theta, dtheta_out]).T])
#                     paramsArray.append({'lambda': lamda, 'fw': fw_val, 'oneOverC': c})

#     return resArray, paramsArray

# def gauss_seidel_method():
#     """Placeholder for Gauss-Seidel method implementation.
    
#     This function is currently not implemented.
#     """
#     pass

# data = [ # ערכי ההתחלה מהמאמר
#     # --- Lambda = 0.5 ---
#     # fw = -1
#     [0.5, -1.0, 1.0, -0.8862, 1.8862, 3.2154],
#     [0.5, -1.0, 2.0, -1.0450, 2.5547, 2.9088],
#     [0.5, -1.0, 5.0, -1.3575, 4.1212, 2.4379],
#     [0.5, -1.0, 8.0, -1.5724, 5.3921, 1.8704],
#     # fw = 0-
#     [0.5, 0.0, 1.0, -1.1020, 1.7474, 2.8250],
#     [0.5, 0.0, 2.0, -1.2495, 2.3479, 2.6049],
#     [0.5, 0.0, 5.0, -1.5503, 3.7996, 2.2380],
#     [0.5, 0.0, 8.0, -1.7610, 4.9990, 2.0340],
#     # fw = 1-
#     [0.5, 1.0, 1.0, -1.3745, 1.6264, 2.4624],
#     [0.5, 1.0, 2.0, -1.5041, 2.1591, 2.3115],
#     [0.5, 1.0, 5.0, -1.7825, 3.4927, 2.0347],
#     [0.5, 1.0, 8.0, -1.9836, 4.6181, 1.8688],

#     # --- Lambda = 2.0 ---
#     # fw = -1
#     [2.0, -1.0, 1.0, -1.6309, 2.1235, 2.2138],
#     [2.0, -1.0, 2.0, -1.9494, 2.9781, 2.0284],
#     [2.0, -1.0, 5.0, -2.5716, 4.9815, 1.7304],
#     [2.0, -1.0, 8.0, -2.9971, 6.6069, 1.5729],
#     # fw = 0, 3.2154
#     [2.0, 0.0, 1.0, -2.0044, 1.9159, 1.8532],
#     [2.0, 0.0, 2.0, -2.2889, 2.6630, 1.7334],
#     [2.0, 0.0, 5.0, -2.8820, 4.4981, 1.5188],
#     [2.0, 0.0, 8.0, -3.2927, 6.0105, 1.3954],
#     # fw = 1-
#     [2.0, 1.0, 1.0, -2.5182, 1.7391, 1.5371],
#     [2.0, 1.0, 2.0, -2.7574, 2.3852, 1.4656],
#     [2.0, 1.0, 5.0, -3.2827, 4.0268, 1.3230],
#     [2.0, 1.0, 8.0, -3.6683, 5.4273, 1.2327]
# ]

# dataOptimized = [ # ערכי ההתחלה מהמאמר
#     # --- Lambda = 0.5 ---
#     # fw = -1
#     [0.5, -1.0, 1.0, -0.8862, 1.8862, 3.2154],
#     [0.5, -1.0, 2.0, -1.0450-0.1, 2.5547-0.1, 2.9088],
#     [0.5, -1.0, 5.0, -1.3575, 4.1212-4, 2.4379],
#     [0.5, -1.0, 8.0, -1.5724, 5.3921-5, 1.8704],
#     # fw = 0
#     [0.5, 0.0, 1.0, -1.1020, 1.7474, 2.8250],
#     [0.5, 0.0, 2.0, -1.2495, 2.3479, 2.6049],
#     [0.5, 0.0, 5.0, -1.5503-2.2, 3.7996-2.2, 2.2380],
#     [0.5, 0.0, 8.0, -1.7610-2.2, 4.9990-2.2, 2.0340],
#     # fw = 1
#     [0.5, 1.0, 1.0, -1.3745, 1.6264, 2.4624],
#     [0.5, 1.0, 2.0, -1.5041, 2.1591, 2.3115],
#     [0.5, 1.0, 5.0, -1.7825-3, 3.4927-3, 2.0347],
#     [0.5, 1.0, 8.0, -1.9836-3, 4.6181-3, 1.8688],

#     # --- Lambda = 2.0 ---
#     # fw = -1
#     [2.0, -1.0, 1.0, -1.6309, 2.1235, 2.2138],
#     [2.0, -1.0, 2.0, -1.9494, 2.9781, 2.0284],
#     [2.0, -1.0, 5.0, -2.5716-0.35, 4.9815-4, 1.7304],
#     [2.0, -1.0, 8.0, -2.9971-0.35, 6.6069-5, 1.5729],
#     # fw = 0, 3.2154
#     [2.0, 0.0, 1.0, -2.0044, 1.9159, 1.8532],
#     [2.0, 0.0, 2.0, -2.2889, 2.6630, 1.7334],
#     [2.0, 0.0, 5.0, -2.8820, 4.4981, 1.5188],
#     [2.0, 0.0, 8.0, -3.2927, 6.0105, 1.3954],
#     # fw = 1
#     [2.0, 1.0, 1.0, -2.5182, 1.7391, 1.5371],
#     [2.0, 1.0, 2.0, -2.7574, 2.3852, 1.4656],
#     [2.0, 1.0, 5.0, -3.2827, 4.0268, 1.3230],
#     [2.0, 1.0, 8.0, -3.6683, 5.4273, 1.2327]
# ]

# # initGuessArray = np.array(data)
# initGuessArray = np.array(dataOptimized)
# # initGuessArray[:, 3:5] -= 0.1
# # initGuessArray[:, 3] += 0.01
# # initGuessArray[:, 4] -= 0.1

# # lambdaArray = np.asarray([0.5, 2])
# # lambdaArray = np.asarray([2])
# lambdaArray = np.asarray([0.5])
# fw = np.asarray([-1, 0, 1])
# # fw = np.asarray([-1])
# # fw = np.asarray([0])
# # fw = np.asarray([1])
# # fw = np.asarray([0, 1])
# # oneOverC = np.asarray([1, 2, 5, 8])
# # oneOverC = np.asarray([1, 2])
# oneOverC = np.asarray([1, 2, 5, 8])
# # oneOverC = np.asarray([2, 5, 8])
# # oneOverC = np.asarray([5, 8])
# # oneOverC = np.asarray([1])

# resArray = []

# # ערכי מטרה באינסוף
# # TARGET_F_TAG = C_PARAM**2
# TARGET_THETA = 0.0

# lambdaLabels = {0.5: '0.5', 2.0: '2.0'}
# cLabels = {1: '1', 2: '2', 5: '5', 8: '8'}
# labels = ['f(eta)', "f '(eta)", 'theta(eta)', "theta '(eta)"]

# # Run both methods
# print("Running Shooting Method...")
# resArray_shooting, paramsArray_shooting = shooting_method(lambdaArray, fw, oneOverC, initGuessArray, TARGET_THETA, EPS=1e-5)
# print("\nRunning Finite Difference Method...")
# resArray_fd, paramsArray_fd = finite_difference_method(lambdaArray, fw, oneOverC, initGuessArray, TARGET_THETA, EPS=1e-4)

# # Flexible plotting function
# def plot_comparison(variable_idx, filter_lambda=None, filter_fw=None, filter_oneOverC=None, 
#                    plot_shooting=True, plot_fd=False):
#     """Plot and compare results from shooting and finite difference methods with filtering options.
    
#     Args:
#         variable_idx (int): Index of variable to plot (0=f, 1=f', 2=theta, 3=theta').
#         filter_lambda (float, optional): Filter results by lambda value (0.5 or 2.0). Defaults to None.
#         filter_fw (float, optional): Filter results by fw value (-1, 0, or 1). Defaults to None.
#         filter_oneOverC (float, optional): Filter results by 1/C value (1, 2, 5, or 8). Defaults to None.
#         plot_shooting (bool, optional): Whether to plot shooting method results. Defaults to True.
#         plot_fd (bool, optional): Whether to plot finite difference results. Defaults to True.
    
#     Returns:
#         None: Displays matplotlib figure.
    
#     Examples:
#         >>> plot_comparison(2)  # Plot all theta for all parameters
#         >>> plot_comparison(3, filter_oneOverC=2)  # Plot theta' only for 1/C=2
#         >>> plot_comparison(2, filter_fw=1, plot_fd=False)  # Plot theta from shooting only, fw=1
#     """
#     plt.figure(figsize=(10, 6))
#     plt.grid()
    
#     # Plot shooting method results
#     if plot_shooting:
#         for i, (run, params) in enumerate(zip(resArray_shooting, paramsArray_shooting)):
#             # Apply filters
#             if filter_lambda is not None and params['lambda'] != filter_lambda:
#                 continue
#             if filter_fw is not None and params['fw'] != filter_fw:
#                 continue
#             if filter_oneOverC is not None and params['oneOverC'] != filter_oneOverC:
#                 continue
            
#             eta_vals = run[0]
#             f_vals = run[1][:, variable_idx]
#             legend_label = f"Shoot: λ={params['lambda']}, fw={params['fw']}, 1/C={params['oneOverC']}"
#             plt.plot(eta_vals, f_vals, label=legend_label, linestyle='-')
    
#     # Plot finite difference results
#     if plot_fd:
#         for i, (run, params) in enumerate(zip(resArray_fd, paramsArray_fd)):
#             # Apply filters
#             if filter_lambda is not None and params['lambda'] != filter_lambda:
#                 continue
#             if filter_fw is not None and params['fw'] != filter_fw:
#                 continue
#             if filter_oneOverC is not None and params['oneOverC'] != filter_oneOverC:
#                 continue
            
#             eta_vals = run[0]
#             f_vals = run[1][:, variable_idx]
#             legend_label = f"FD: λ={params['lambda']}, fw={params['fw']}, 1/C={params['oneOverC']}"
#             plt.plot(eta_vals, f_vals, label=legend_label, linestyle='--')
    
#     plt.xlabel('Eta')
#     plt.ylabel(labels[variable_idx])
#     plt.title(labels[variable_idx])
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     # plt.tight_layout()

# # Example plots - uncomment the ones you want:

# # # Plot all theta (variable_idx=2) for all fw values
# plot_comparison(2, filter_lambda=None, filter_fw=None, filter_oneOverC=None)

# # Plot all f' (variable_idx=1) for all runs
# plot_comparison(1)

# # Plot theta' (variable_idx=3) for oneOverC=2
# plot_comparison(3, filter_oneOverC=2)

# # Plot theta for fw=1 only
# # plot_comparison(2, filter_fw=1)

# # Plot f for lambda=0.5, fw=0
# plot_comparison(0, filter_lambda=None, filter_fw=None)

# plt.show()







# קוד שעובד טוב אך לא מתכנס בכל הערכים
# import numpy as np
# import matplotlib.pyplot as plt

# # --- הגדרת הפרמטרים מהתמונה ---
# lambda_vals = [0.5, 2.0]
# fw_vals = [-1.0, 0.0, 1.0]
# one_over_c_vals = [1.0, 2.0, 5.0, 8.0]

# # --- פרמטרים נומריים ---
# BASE_ETA = 10.0      # בסיס לאינסוף (הגדלתי מעט מ-8)
# H = 0.02            # צעד קטן יותר לדיוק
# TOL = 1e-4          # סובלנות שגיאה
# MAX_ITER = 40       # יותר איטרציות
# DELTA = 1e-4        # צעד לנגזרת נומרית

# def get_dynamic_eta(fw, one_over_c):
#     """
#     קביעת טווח האינטגרציה.
#     בהזרקה (fw=-1) ובקונבקציה טבעית (1/C גדול), שכבת הגבול עבה מאוד.
#     """
#     eta = BASE_ETA
    
#     # פקטור עבור הזרקה
#     if fw == -1.0:
#         eta *= 3.0
    
#     # פקטור עבור קונבקציה טבעית חזקה (C קטן)
#     if one_over_c >= 5.0:
#         eta *= 1.5
        
#     return eta

# def get_derivatives(eta, y, lam):
#     """ y = [f, f', theta, theta'] """
#     f, df, theta, dtheta = y
    
#     # משוואה 8
#     ddf = -((lam - 2)/3 * eta * dtheta + lam * theta)
#     # משוואה 9
#     ddtheta = lam * df * theta - (lam + 1)/3 * f * dtheta
    
#     return np.array([df, ddf, dtheta, ddtheta])

# def rk4_step(eta, y, h, lam):
#     """ צעד RK4 בודד """
#     k1 = h * get_derivatives(eta, y, lam)
#     k2 = h * get_derivatives(eta + 0.5*h, y + 0.5*k1, lam)
#     k3 = h * get_derivatives(eta + 0.5*h, y + 0.5*k2, lam)
#     k4 = h * get_derivatives(eta + h, y + k3, lam)
#     return y + (k1 + 2*k2 + 2*k3 + k4) / 6.0

# def integrate(f_start, guess_df0, guess_dtheta0, lam, eta_inf):
#     """ ביצוע אינטגרציה מלאה """
#     steps = int(np.ceil(eta_inf / H))
    
#     # אתחול
#     y = np.array([f_start, guess_df0, 1.0, guess_dtheta0])
    
#     history_f = [y[0]]
#     history_df = [y[1]]
#     history_theta = [y[2]]
#     eta_vals = [0.0]
    
#     curr_eta = 0.0
    
#     # לולאת אינטגרציה
#     for _ in range(steps):
#         y = rk4_step(curr_eta, y, H, lam)
#         curr_eta += H
        
#         # בדיקת "פיצוץ" (NaN או מספרים עצומים)
#         if np.isnan(y).any() or np.abs(y[1]) > 1e5:
#             return None, None, None # סימן לכישלון באינטגרציה
            
#         history_f.append(y[0])
#         history_df.append(y[1])
#         history_theta.append(y[2])
#         eta_vals.append(curr_eta)
        
#     return y, np.array(eta_vals), np.array([history_f, history_df, history_theta])

# def solve_case(lam, fw, one_over_c):
#     # חישוב פרמטרים
#     current_eta_inf = get_dynamic_eta(fw, one_over_c)
#     c_val = 1.0 / one_over_c
    
#     f_start = fw * (c_val**(-1/3))
#     target_f_prime = c_val**(2/3)
#     target_theta = 0.0
    
#     # === ניחוש התחלתי חכם ===
#     # זה החלק הקריטי למניעת התבדרות
#     guess_df0 = target_f_prime # ברירת מחדל: השיפוע באינסוף
#     guess_dtheta0 = -1.0
    
#     if fw == -1.0: # בהזרקה, הנגזרות על הקיר קטנות יותר
#         guess_df0 = target_f_prime * 0.5 
#         guess_dtheta0 = -0.5
#     elif fw == 1.0: # ביניקה, הן גדולות יותר
#         guess_df0 = target_f_prime * 1.5
#         guess_dtheta0 = -2.0

#     converged = False
#     best_history = None
#     best_eta = None
    
#     # לולאת ניוטון-רפסון
#     for i in range(MAX_ITER):
#         # 1. הרצה נוכחית
#         y_end, eta_arr, hist = integrate(f_start, guess_df0, guess_dtheta0, lam, current_eta_inf)
        
#         if y_end is None: # טיפול בהתבדרות של RK4
#             # אם האינטגרציה נכשלה, נקטין את הניחושים (נסיגה)
#             guess_df0 *= 0.8
#             guess_dtheta0 *= 0.8
#             continue

#         err_fp = y_end[1] - target_f_prime
#         err_t = y_end[2] - target_theta
        
#         # בדיקת התכנסות
#         if np.sqrt(err_fp**2 + err_t**2) < TOL:
#             converged = True
#             best_history = hist
#             best_eta = eta_arr
#             break
            
#         # 2. חישוב יעקוביאן
#         # הערה: אם ההפרעה גורמת לפיצוץ, הנגזרת תהיה לא תקינה. נוסיף הגנה.
#         y_dA, _, _ = integrate(f_start, guess_df0 + DELTA, guess_dtheta0, lam, current_eta_inf)
#         if y_dA is None: y_dA = y_end # Fallback פרימיטיבי למניעת קריסה
            
#         y_dB, _, _ = integrate(f_start, guess_df0, guess_dtheta0 + DELTA, lam, current_eta_inf)
#         if y_dB is None: y_dB = y_end

#         J11 = (y_dA[1] - y_end[1]) / DELTA
#         J21 = (y_dA[2] - y_end[2]) / DELTA
#         J12 = (y_dB[1] - y_end[1]) / DELTA
#         J22 = (y_dB[2] - y_end[2]) / DELTA
        
#         det = J11*J22 - J12*J21
        
#         if abs(det) < 1e-12:
#             # במקרה של דטרמיננטה אפסית, נבצע שינוי אקראי קטן לניחוש כדי לצאת מהתקיעות
#             guess_df0 += np.random.randn() * 0.1
#             continue
            
#         d_df0 = -(J22 * err_fp - J12 * err_t) / det
#         d_dtheta0 = -(J11 * err_t - J21 * err_fp) / det
        
#         # === ריסון (Damping/Clamping) ===
#         # לא ניתן לתיקון להיות גדול מדי בצעד אחד
#         d_df0 = np.clip(d_df0, -0.5, 0.5)
#         d_dtheta0 = np.clip(d_dtheta0, -0.5, 0.5)
        
#         # פקטור רלקסציה (מאט את ההתכנסות אבל מונע התבדרות)
#         relaxation = 0.7 
#         guess_df0 += d_df0 * relaxation
#         guess_dtheta0 += d_dtheta0 * relaxation
        
#     return best_eta, best_history, converged

# # --- הרצה והצגה ---
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# print(f"{'L':<4} {'fw':<4} {'1/C':<4} | {'Status'}")
# print("-" * 30)

# line_styles = ['-', '--', '-.', ':']
# colors = plt.cm.turbo(np.linspace(0, 1, 12)) # פלטת צבעים עשירה יותר

# idx = 0
# for lam in lambda_vals:
#     for fw in fw_vals:
#         for one_over_c in one_over_c_vals:
            
#             eta, hist, success = solve_case(lam, fw, one_over_c)
            
#             status = "OK" if success else "FAIL"
#             print(f"{lam:<4} {int(fw):<4} {int(one_over_c):<4} | {status}")
            
#             if success:
#                 f = hist[0]
#                 f_prime = hist[1]
#                 theta = hist[2]
                
#                 # בחירת סגנון
#                 ls = line_styles[idx % 4]
#                 color = colors[idx % 12]
                
#                 # אנו נציג רק תווית חלקית כדי לא להעמיס
#                 lbl = None
#                 if idx % 3 == 0: 
#                     lbl = f"L={lam},fw={int(fw)},1/C={int(one_over_c)}"
                
#                 ax1.plot(eta, theta, ls=ls, color=color, lw=1.2, label=lbl, alpha=0.8)
#                 ax2.plot(eta, f_prime, ls=ls, color=color, lw=1.2, alpha=0.8)
#                 ax3.plot(eta, f, ls=ls, color=color, lw=1.2, alpha=0.8)
                
#                 idx += 1

# # עיצוב
# ax1.set_title(r"Temperature ($\theta$)")
# ax1.set_xlabel(r"$\eta$")
# ax1.axhline(0, color='k', lw=1)
# ax1.grid(True, alpha=0.3)
# ax1.legend(loc='upper right', fontsize='x-small')

# ax2.set_title(r"Velocity ($f'$)")
# ax2.set_xlabel(r"$\eta$")
# ax2.grid(True, alpha=0.3)

# ax3.set_title(r"Stream Function ($f$)")
# ax3.set_xlabel(r"$\eta$")
# ax3.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import sys

# --- הגדרת הפרמטרים ---
lambda_vals = [0.5, 2.0]
fw_vals = [-1.0, 0.0, 1.0]
one_over_c_vals = [1.0, 2.0, 5.0, 8.0]

# --- פרמטרים נומריים ---
H = 0.02            # צעד אינטגרציה
TOL = 1e-4          # סובלנות
DELTA = 1e-4        # לחישוב נגזרות

# ==========================================
# 1. מנוע RK4 (ידני)
# ==========================================
def get_derivatives(eta, y, lam):
    f, df, theta, dtheta = y
    ddf = -((lam - 2)/3 * eta * dtheta + lam * theta)
    ddtheta = lam * df * theta - (lam + 1)/3 * f * dtheta
    return np.array([df, ddf, dtheta, ddtheta])

def rk4_step(eta, y, h, lam):
    k1 = h * get_derivatives(eta, y, lam)
    k2 = h * get_derivatives(eta + 0.5*h, y + 0.5*k1, lam)
    k3 = h * get_derivatives(eta + 0.5*h, y + 0.5*k2, lam)
    k4 = h * get_derivatives(eta + h, y + k3, lam)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6.0

def integrate(f_start, df0, dtheta0, lam, eta_inf):
    steps = int(np.ceil(eta_inf / H))
    y = np.array([f_start, df0, 1.0, dtheta0])
    
    hist_f, hist_df, hist_theta = [y[0]], [y[1]], [y[2]]
    eta_vals = [0.0]
    
    curr_eta = 0.0
    valid = True
    
    for _ in range(steps):
        y = rk4_step(curr_eta, y, H, lam)
        curr_eta += H
        
        if np.isnan(y).any() or np.abs(y[1]) > 500:
            valid = False
            break
            
        hist_f.append(y[0])
        hist_df.append(y[1])
        hist_theta.append(y[2])
        eta_vals.append(curr_eta)
        
    return y, np.array(eta_vals), np.array([hist_f, hist_df, hist_theta]), valid

# ==========================================
# 2. לוגיקה חכמה לפתרון (Grid Search + Newton)
# ==========================================

def get_smart_eta(fw):
    if fw == -1.0: return 10.0  
    if fw == 0.0:  return 10.0
    return 5.0                 

def _grid_search_initial_guess(
    *,
    lam: float,
    f_start: float,
    eta_inf: float,
    target_fp: float,
    target_theta: float,
    df_min: float,
    df_max: float,
    dtheta_min: float,
    dtheta_max: float,
    n_df: int,
    n_dtheta: int,
):
    best_guess = None
    best_err = np.inf

    scan_df = np.linspace(df_min, df_max, n_df)
    scan_dtheta = np.linspace(dtheta_min, dtheta_max, n_dtheta)

    for g_df in scan_df:
        for g_dt in scan_dtheta:
            y_end, _, _, valid = integrate(f_start, g_df, g_dt, lam, eta_inf)
            if not valid:
                continue

            err_fp = y_end[1] - target_fp
            err_t = y_end[2] - target_theta
            err = float(np.hypot(err_fp, err_t))

            if err < best_err:
                best_err = err
                best_guess = (float(g_df), float(g_dt))

    return best_guess, best_err

def _newton_solve(
    *,
    lam: float,
    f_start: float,
    eta_inf: float,
    target_fp: float,
    target_theta: float,
    df0: float,
    dtheta0: float,
    max_iter: int = 40,
):
    curr_df, curr_dt = float(df0), float(dtheta0)

    for _ in range(max_iter):
        y_end, eta_arr, hist, valid = integrate(f_start, curr_df, curr_dt, lam, eta_inf)
        if not valid:
            # back off to regain validity
            curr_df *= 0.9
            curr_dt *= 0.9
            continue

        err_fp = float(y_end[1] - target_fp)
        err_t = float(y_end[2] - target_theta)
        err_norm = float(np.hypot(err_fp, err_t))

        if err_norm < TOL:
            return (eta_arr, hist), True, curr_df, curr_dt

        # Finite-difference Jacobian
        y_A, _, _, vA = integrate(f_start, curr_df + DELTA, curr_dt, lam, eta_inf)
        y_B, _, _, vB = integrate(f_start, curr_df, curr_dt + DELTA, lam, eta_inf)
        if (not vA) or (not vB):
            # small perturbation to escape invalid region
            curr_df = max(0.0, curr_df + 0.05)
            continue

        J = np.array(
            [
                [(y_A[1] - y_end[1]) / DELTA, (y_B[1] - y_end[1]) / DELTA],
                [(y_A[2] - y_end[2]) / DELTA, (y_B[2] - y_end[2]) / DELTA],
            ],
            dtype=float,
        )
        rhs = np.array([err_fp, err_t], dtype=float)

        try:
            step = -np.linalg.solve(J, rhs)
        except np.linalg.LinAlgError:
            curr_df = max(0.0, curr_df + 0.05)
            continue

        # Clamp step sizes to prevent blow-ups
        step[0] = float(np.clip(step[0], -0.75, 0.75))
        step[1] = float(np.clip(step[1], -1.5, 1.5))

        # Backtracking line search: accept only if error decreases
        alpha = 1.0
        accepted = False
        for _ls in range(8):
            trial_df = max(0.0, curr_df + alpha * step[0])
            trial_dt = curr_dt + alpha * step[1]
            y_trial, _, _, valid_trial = integrate(f_start, trial_df, trial_dt, lam, eta_inf)

            if not valid_trial:
                alpha *= 0.5
                continue

            trial_err = float(np.hypot(y_trial[1] - target_fp, y_trial[2] - target_theta))
            if trial_err < err_norm:
                curr_df, curr_dt = trial_df, trial_dt
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            # If Newton direction doesn't help, nudge mildly and keep going
            curr_df = max(0.0, curr_df + 0.02)

    return None, False, curr_df, curr_dt

def solve_single_case(lam, fw, one_over_c, *, seed_guess=None):
    # input is 1/C, so C = 1/(1/C)
    c_val = 1.0 / float(one_over_c)
    f_start = float(fw) * (c_val ** (-1.0 / 3.0))
    target_fp = float(c_val ** (2.0 / 3.0))
    target_theta = 0.0

    # thicker BL for injection and for stronger natural convection (large 1/C)
    current_eta = float(get_smart_eta(fw))
    if fw == -1.0:
        current_eta *= 1.0 + 0.08 * float(one_over_c)
    else:
        current_eta *= 1.0 + 0.03 * float(one_over_c)

    # ===== Stage A: grid search to get a good Newton seed =====
    # If we have a continuation guess from a nearby case, search locally around it.
    if seed_guess is not None:
        seed_df, seed_dt = seed_guess
        df_min = max(0.0, seed_df - 1.0)
        df_max = seed_df + 1.0
        dt_min = seed_dt - 3.0
        dt_max = min(-1e-6, seed_dt + 3.0)
        n_df, n_dt = 17, 33
    else:
        # Generic coarse ranges (wide enough to cover all your cases)
        df_min = 0.0
        df_max = max(2.0, 4.0 * target_fp + 2.0)
        dt_min = -30.0
        dt_max = -1e-3
        n_df, n_dt = 21, 61

    best_guess, best_err = _grid_search_initial_guess(
        lam=lam,
        f_start=f_start,
        eta_inf=current_eta,
        target_fp=target_fp,
        target_theta=target_theta,
        df_min=df_min,
        df_max=df_max,
        dtheta_min=dt_min,
        dtheta_max=dt_max,
        n_df=n_df,
        n_dtheta=n_dt,
    )

    # Refinement pass around the best coarse guess (helps Newton a lot)
    if best_guess is not None:
        bdf, bdt = best_guess
        best_guess2, _ = _grid_search_initial_guess(
            lam=lam,
            f_start=f_start,
            eta_inf=current_eta,
            target_fp=target_fp,
            target_theta=target_theta,
            df_min=max(0.0, bdf - 0.4),
            df_max=bdf + 0.4,
            dtheta_min=bdt - 1.0,
            dtheta_max=min(-1e-6, bdt + 1.0),
            n_df=17,
            n_dtheta=25,
        )
        if best_guess2 is not None:
            best_guess = best_guess2
    else:
        # last-resort fallback
        best_guess = (target_fp, -1.0)
        best_err = np.inf

    # ===== Stage B: damped Newton solve =====
    res, success, final_df, final_dt = _newton_solve(
        lam=lam,
        f_start=f_start,
        eta_inf=current_eta,
        target_fp=target_fp,
        target_theta=target_theta,
        df0=best_guess[0],
        dtheta0=best_guess[1],
        max_iter=50,
    )

    # If Newton failed, try one more time with expanded eta and a wider grid.
    if (not success) and (seed_guess is None) and (best_err > 1e-2):
        eta2 = current_eta * 1.5
        g2, _ = _grid_search_initial_guess(
            lam=lam,
            f_start=f_start,
            eta_inf=eta2,
            target_fp=target_fp,
            target_theta=target_theta,
            df_min=0.0,
            df_max=max(2.0, 6.0 * target_fp + 3.0),
            dtheta_min=-40.0,
            dtheta_max=-1e-3,
            n_df=25,
            n_dtheta=71,
        )
        if g2 is not None:
            res, success, final_df, final_dt = _newton_solve(
                lam=lam,
                f_start=f_start,
                eta_inf=eta2,
                target_fp=target_fp,
                target_theta=target_theta,
                df0=g2[0],
                dtheta0=g2[1],
                max_iter=60,
            )

    return res, success, final_df, final_dt

# ==========================================
# 3. הרצה והצגה עם מקרא (Legend)
# ==========================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))

print(f"{'Idx':<4} {'L':<4} {'fw':<4} {'1/C':<4} | {'Status'}")
print("-" * 40)

count = 0
colors = plt.cm.jet(np.linspace(0, 1, 24)) 

for lam in lambda_vals:
    for fw in fw_vals:
        continuation_guess = None
        for one_over_c in one_over_c_vals:
            count += 1

            res, success, final_df, final_dt = solve_single_case(
                lam, fw, one_over_c, seed_guess=continuation_guess
            )
            
            status = "OK" if success else "FAIL"
            print(f"{count:<4} {lam:<4} {int(fw):<4} {int(one_over_c):<4} | {status}")
            
            if success:
                continuation_guess = (final_df, final_dt)
                eta_vals, hist = res
                theta = hist[2]
                f_prime = hist[1]
                f_val = hist[0]
                
                # יצירת תווית למקרא
                label_str = f"$\lambda={lam}, f_w={int(fw)}, 1/C={int(one_over_c)}$"
                ls = '-' if lam == 0.5 else '--'
                
                ax1.plot(eta_vals, theta, ls=ls, color=colors[count-1], lw=1.2, alpha=0.8, label=label_str)
                ax2.plot(eta_vals, f_prime, ls=ls, color=colors[count-1], lw=1.2, alpha=0.8, label=label_str)
                ax3.plot(eta_vals, f_val, ls=ls, color=colors[count-1], lw=1.2, alpha=0.8, label=label_str)

# עיצוב גרפים
ax1.set_title(r"Temperature $\theta(\eta)$")
ax1.set_xlabel(r"$\eta$")
ax1.set_ylabel(r"$\theta$")
ax1.axhline(0, color='k', lw=1)
ax1.grid(True, alpha=0.3)

# יצירת המקרא המשותף בצד ימין (מחוץ לגרפים)
# אנו לוקחים את ה-handles מאחד הגרפים
handles, labels = ax3.get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.08, 0.5), fontsize='small', title="Parameters")

ax2.set_title(r"Velocity $f'(\eta)$")
ax2.set_xlabel(r"$\eta$")
ax2.set_ylabel(r"$f'$")
ax2.grid(True, alpha=0.3)

ax3.set_title(r"Stream Function $f(\eta)$")
ax3.set_xlabel(r"$\eta$")
ax3.set_ylabel(r"$f$")
ax3.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 0.88, 1]) # השארת מקום למקרא
plt.show()
