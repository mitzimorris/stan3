#ifndef STAN3_C_API_H
#define STAN3_C_API_H

#include <stddef.h>

/* Export macro for shared library visibility */
#if defined(_WIN32) || defined(__CYGWIN__)
  #ifdef STAN3_BUILDING_LIBRARY
    #define STAN3_API __declspec(dllexport)
  #else
    #define STAN3_API __declspec(dllimport)
  #endif
#else
  #if defined(STAN3_BUILDING_LIBRARY) && defined(__GNUC__)
    #define STAN3_API __attribute__((visibility("default")))
  #else
    #define STAN3_API
  #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes */
#define STAN3_SUCCESS 0
#define STAN3_ERROR_PARSING 1
#define STAN3_ERROR_MODEL_LOAD 2
#define STAN3_ERROR_SAMPLING 3
#define STAN3_ERROR_INVALID_ARGS 4
#define STAN3_ERROR_RUNTIME 5

/* Load a Stan model using the provided command-line arguments
 * 
 * @param argc Number of arguments
 * @param argv Array of argument strings
 * @param error_message Buffer to store error message on failure
 * @param error_message_size Size of error message buffer
 * @return STAN3_SUCCESS on success, error code on failure
 */
STAN3_API int stan3_load_model(int argc, char** argv, 
                               char* error_message, size_t error_message_size);

/* Run samplers on the loaded model using command-line arguments
 * 
 * @param argc Number of arguments  
 * @param argv Array of argument strings
 * @param error_message Buffer to store error message on failure
 * @param error_message_size Size of error message buffer
 * @return STAN3_SUCCESS on success, error code on failure
 */
STAN3_API int stan3_run_samplers(int argc, char** argv,
                                 char* error_message, size_t error_message_size);

/* Get the name of the currently loaded model
 * 
 * @return Model name string, or NULL if no model loaded
 */
STAN3_API const char* stan3_get_model_name(void);

/* Check if a model is currently loaded
 * 
 * @return 1 if model loaded, 0 otherwise
 */
STAN3_API int stan3_is_model_loaded(void);

/* Get the last error message (thread-local storage)
 * 
 * @return Error message string, or NULL if no error
 */
STAN3_API const char* stan3_get_last_error(void);

/* Clear the last error message
 */
STAN3_API void stan3_clear_error(void);

#ifdef __cplusplus
}
#endif

#endif  /* STAN3_C_API_H */
