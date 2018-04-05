import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import math
import sys
import copy as cp
import matplotlib.pyplot as plt
import numbers
import collections
from sklearn import metrics
import dill

### This module contains functions that accomplish the tasks of determining the reliability
### of questions and analysts, as well as the task of combining the results of analysts, and of
### pickling and outputing a function that can be used to do the combination



ALL_ANALYSTS_EXCEPT = 'all_analysts_except_'
THE_OTHER_ANALYSTS = 'the_other_analysts'



#########
### 
###   API TO EVALUATE ANALYST AGREEMENT/ACCURACY AND QUESTION DIFFICULTIES
###
#########


def rate_analysts_against_eachother(scoresheet_dataframe, analysts_to_rate, questions_to_rate_over,
                                    rating_aggregation_method=np.mean, analyst_id_column_name='analyst',
                                    subject_id_column_name='submission'):

    ''' This function is what you should use to determine the reliabilitiy of analysts over some questions, each compared to the rest
    
    
        Functionally, it will perform the rating method on every analyst and for every question. Then it will
    aggregate these ratings across questions using an aggregation method of your choice.

    inputs:
       scoresheet_dataframe: of raw data, in format of one row per subject and per question, and one column per analyst
       analysts_to_rate: list of analysts to include in the rating
       questions_to_rate_over: list of questions to calculate reliabilities for and aggregate
       rating_aggregation_method: function to combine individual rating calculations
    '''

    #make sure we're dealing with lists in case the user passed some other iterable
    analysts_to_rate = list(analysts_to_rate)
    questions_to_rate_over = list(questions_to_rate_over)

    supported_rating_methods = ['percent_agreement', 'cohen_kappa']  #note: weighted_cohen_kappa, weighted_fleiss_kappa not supported for analyst rating

    #make sure all analysts the user wants to rate actually exist in the passed dataframe
    if analysts_to_rate is None or len(analysts_to_rate)<2:
            raise ValueError("To compare analyst to analyst you need to include at least two of them")
    present_analysts = set(scoresheet_dataframe[analyst_id_column_name])
    for analyst in analysts_to_rate:
        if analyst not in present_analysts:
            raise ValueError("Couldn't find any data for analyst "+analyst+". Make sure the input scoresheet dataframe is complete")

    #every analyst in the scoresheet other than the ones specified by the user will be excluded
    analysts_to_exclude = present_analysts - set(analysts_to_rate)
    
    #pivot the scoresheet dataset into what is called a "comparison dataframe" format
    analyst_comparison_dataframe = _create_analyst_comparison_dataframe(scoresheet_dataframe, questions_to_include_in_comparison=questions_to_rate_over, 
                analyst_id_column_name=analyst_id_column_name, reference_analysts=[], exclude_these_analysts=analysts_to_exclude, subject_id_column_name= subject_id_column_name, indeterminate_answer_code=9, debug=False)
    
    
    #compute every supported agreement metric, one at a time
    analyst_reliability_output = {}
    analyst_question_reliability_output = {}

    for rating_method in supported_rating_methods:
        
        #call on some helper functions to run comparisons on a question and analyst levels        
        analyst_question_reliabilities = get_separate_reliabilities_for_all_analysts_and_questions(analyst_comparison_dataframe, calculation_to_do=rating_method,
                questions_to_include=questions_to_rate_over, reference_column_name=THE_OTHER_ANALYSTS, debug=False)
        analyst_question_reliabilities = analyst_question_reliabilities.sort_values(analyst_id_column_name)
        analyst_reliabilities = analyst_question_reliabilities.groupby(analyst_id_column_name).apply(rating_aggregation_method)
        analyst_reliabilities = analyst_reliabilities.sort_index()

        #put the question-specific agreement scores into the output data structure
        analyst_question_reliability_output[rating_method] = {}
        for question in questions_to_rate_over:
            analyst_question_reliability_output[rating_method][question] = [0.0] * len(analysts_to_rate)
        for analyst, question, reliability in zip(analyst_question_reliabilities[analyst_id_column_name], analyst_question_reliabilities['question'], analyst_question_reliabilities['reliability'].values):
            analyst_question_reliability_output[rating_method][question][analysts_to_rate.index(analyst.replace('analyst_',''))] = reliability
            
        #put the question-aggregated agreement scores into the output data structure
        analyst_reliability_output[rating_method] = [0.0] * len(analysts_to_rate)
        for analyst, reliability in zip(analyst_reliabilities.index, analyst_reliabilities['reliability'].values):
            analyst_reliability_output[rating_method][analysts_to_rate.index(analyst.replace('analyst_',''))] = reliability



    #now let's evaluate the 'coverage' rate per question per analyst
    analyst_question_coverage_output = {}
    for question in questions_to_rate_over:
        analyst_question_coverage_output[question] = []
        for analyst in analysts_to_rate:
            subset = analyst_comparison_dataframe[analyst_comparison_dataframe['question']==question]
            analyst_question_coverage = rating_aggregation_method(subset['determinate_answer_by_analyst_'+analyst].values)
            analyst_question_coverage_output[question].append(analyst_question_coverage)

    #now let's evaluate the 'coverage' rate per analyst over all questions
    analyst_coverage_output = []
    coverage_df = analyst_comparison_dataframe.groupby('question').apply(rating_aggregation_method)
    for analyst in analysts_to_rate:
        analyst_overall_coverage = rating_aggregation_method(coverage_df['determinate_answer_by_analyst_'+analyst].values)
        analyst_coverage_output.append(analyst_overall_coverage)
        
    
    #prepare the agreement over all questions as a separate Dataframe
    all_Qs_output_dictionary = {analyst_id_column_name:analysts_to_rate, 'coverage':analyst_coverage_output}
    for rating_method in supported_rating_methods:
        all_Qs_output_dictionary['agreement_using_'+rating_method] = analyst_reliability_output[rating_method]

    #prepare the agreement per question as a separate Dataframe
    per_Q_output_dictionary = {analyst_id_column_name:analysts_to_rate}
    for question in questions_to_rate_over:
        per_Q_output_dictionary['coverage_on_'+question] = analyst_question_coverage_output[question]
        for rating_method in supported_rating_methods:
            per_Q_output_dictionary['agreement_on_'+question+'_using_'+rating_method] = analyst_question_reliability_output[rating_method][question]


    #format the outputs into a dictionary of DataFrame to return to caller  
    output_dict = {}
    output_dict['agreement_over_all_questions'] = pd.DataFrame(all_Qs_output_dictionary)     
    output_dict['agreement_per_question'] = pd.DataFrame(per_Q_output_dictionary)
    return output_dict



def rate_analysts_against_reference(scoresheet_dataframe, analysts_to_rate, reference, questions_to_rate_over, rating_aggregation_method=np.mean, analyst_id_column_name='analyst', subject_id_column_name='submission'):

    ''' This function is what you should use to determine the reliabilitiy of analysts over some questions, each compared to a reference ground truth
    
    
        Functionally, it will perform the rating method on every analyst and for every question. Then it will
    aggregate these ratings across questions using an aggregation method of your choice.

    inputs:
       scoresheet_dataframe: of raw data, in format of one row per subject and per question, and one column per analyst
       analysts_to_rate: list of analysts to include in the rating, other than the reference
       questions_to_rate_over: list of questions to calculate reliabilities for and aggregate
       reference: reference analyst or instrument to consider as ground truth
       rating_aggregation_method: function to combine individual rating calculations
    '''


    #make sure we're dealing with lists in case the user passed some other iterable
    analysts_to_rate = list(analysts_to_rate)
    questions_to_rate_over = list(questions_to_rate_over)


    supported_rating_methods = ['percent_agreement', 'cohen_kappa'] #weighted_cohen_kappa currently turned off, and fleiss_kappa/weighted_fleiss_kappa not supported for analyst rating.

    #make sure all analysts the user wants to rate actually exist in the passed dataframe
    if analysts_to_rate is None or len(analysts_to_rate)<2:
            raise ValueError("To compare analyst to analyst you need to include at least two of them")
    present_analysts = set(scoresheet_dataframe[analyst_id_column_name])
    for analyst in analysts_to_rate:
        if analyst not in present_analysts:
            raise ValueError("Couldn't find any data for analyst "+analyst+". Make sure the input scoresheet dataframe is complete")
   
    #make sure the reference analyst exists in the passed dataframe
    if reference is None:
            raise ValueError("To compare analyst to reference you need to specify a valid analyst or instrument as reference")
    present_analysts = set(scoresheet_dataframe[analyst_id_column_name])
    if reference not in present_analysts:
            raise ValueError("Couldn't find any data for reference "+reference+". Make sure the input scoresheet dataframe is complete")

    #an analyst cannot be chosen as both a reference and as a rated analyst
    if reference in analysts_to_rate:
            raise ValueError("Cannot rate analyst "+reference+" while also using him/her as a reference")

    #every analyst in the scoresheet other than the ones specified by the user will be excluded
    analysts_to_exclude = present_analysts - set(analysts_to_rate+[reference]) 
    
    
    #pivot the scoresheet dataset into what is called a "comparison dataframe" format
    analyst_comparison_dataframe = _create_analyst_comparison_dataframe(scoresheet_dataframe, questions_to_include_in_comparison=questions_to_rate_over, 
                analyst_id_column_name=analyst_id_column_name, reference_analysts=reference, exclude_these_analysts=analysts_to_exclude, subject_id_column_name= subject_id_column_name, indeterminate_answer_code=9, debug=False)
    
    
    
    #compute every supported agreement metric, one at a time
    analyst_reliability_output = {}
    analyst_question_reliability_output = {}

    for rating_method in supported_rating_methods:
        
        #call on some helper functions to run comparisons on a question and analyst levels        
        analyst_question_reliabilities = get_separate_reliabilities_for_all_analysts_and_questions(analyst_comparison_dataframe, calculation_to_do=rating_method,
                questions_to_include=questions_to_rate_over, reference_column_name=reference, debug=False)
        analyst_question_reliabilities = analyst_question_reliabilities.sort_values(analyst_id_column_name)
                
        analyst_reliabilities = analyst_question_reliabilities.groupby(analyst_id_column_name).apply(rating_aggregation_method)
        analyst_reliabilities = analyst_reliabilities.sort_index()

        #put the question-specific agreement scores into the output data structure
        analyst_question_reliability_output[rating_method] = {}
        for question in questions_to_rate_over:
            analyst_question_reliability_output[rating_method][question] = [0.0] * len(analysts_to_rate)
        for analyst, question, reliability in zip(analyst_question_reliabilities[analyst_id_column_name], analyst_question_reliabilities['question'], analyst_question_reliabilities['reliability'].values):
            analyst_question_reliability_output[rating_method][question][analysts_to_rate.index(analyst.replace('analyst_',''))] = reliability
            
        #put the question-aggregated agreement scores into the output data structure
        analyst_reliability_output[rating_method] = [0.0] * len(analysts_to_rate)
        for analyst, reliability in zip(analyst_reliabilities.index, analyst_reliabilities['reliability'].values):
            analyst_reliability_output[rating_method][analysts_to_rate.index(analyst.replace('analyst_',''))] = reliability



    #now let's evaluate the 'coverage' rate per question per analyst
    analyst_question_coverage_output = {}
    for question in questions_to_rate_over:
        analyst_question_coverage_output[question] = []
        for analyst in analysts_to_rate:
            subset = analyst_comparison_dataframe[analyst_comparison_dataframe['question']==question]
            analyst_question_coverage = rating_aggregation_method(subset['determinate_answer_by_analyst_'+analyst].values)
            analyst_question_coverage_output[question].append(analyst_question_coverage)

    #now let's evaluate the 'coverage' rate per analyst over all questions
    analyst_coverage_output = []
    coverage_df = analyst_comparison_dataframe.groupby('question').apply(rating_aggregation_method)
    for analyst in analysts_to_rate:
        analyst_overall_coverage = rating_aggregation_method(coverage_df['determinate_answer_by_analyst_'+analyst].values)
        analyst_coverage_output.append(analyst_overall_coverage)
        
    
    #prepare the agreement over all questions as a separate Dataframe
    all_Qs_output_dictionary = {analyst_id_column_name:analysts_to_rate, 'coverage':analyst_coverage_output}
    for rating_method in supported_rating_methods:
        all_Qs_output_dictionary['agreement_with_'+reference+'_using_'+rating_method] = analyst_reliability_output[rating_method]

    #prepare the agreement per question as a separate Dataframe
    per_Q_output_dictionary = {analyst_id_column_name:analysts_to_rate}
    for question in questions_to_rate_over:
        per_Q_output_dictionary['coverage_on_'+question] = analyst_question_coverage_output[question]
        for rating_method in supported_rating_methods:
            per_Q_output_dictionary['agreement_with_'+reference+'_on_'+question+'_using_'+rating_method] = analyst_question_reliability_output[rating_method][question]


    #format the outputs into a dictionary of DataFrame to return to caller  
    output_dict = {}
    output_dict['agreement_over_all_questions'] = pd.DataFrame(all_Qs_output_dictionary)     
    output_dict['agreement_per_question'] = pd.DataFrame(per_Q_output_dictionary)
    return output_dict



def rate_questions_on_analyst_agreement_with_eachother(scoresheet_dataframe, questions_to_rate, analysts_to_rate_over, rating_aggregation_method=np.mean, analyst_id_column_name='analyst', subject_id_column_name='submission'):


    ''' This function is what you should use to determine the analyst agreement with ground truth on a question by question basis
    
    Functionally, it will perform the rating method on every analyst and for every question. Then it will
    aggregate these ratings across analysts using an aggregation method of your choice.

    inputs:
       scoresheet_dataframe: of raw data, in format of one row per subject and per question, and one column per analyst
       questions_to_rate: list of questions to rate
       analysts_to_rate_over: list of analysts to include in the rating calculations
       rating_aggregation_method: function to combine individual agreement calculations
    '''

    #make sure we're dealing with lists in case the user passed some other iterable
    questions_to_rate = list(questions_to_rate)
    analysts_to_rate_over = list(analysts_to_rate_over)


    supported_rating_methods = ['percent_agreement', 'cohen_kappa'] #weighted_cohen_kappa, fleiss_kappa, and weighted_fleiss_kappa currently turned off

    #make sure all analysts the user wants to rate over actually exist in the passed dataframe
    if analysts_to_rate_over is None or len(analysts_to_rate_over)<1:
            raise ValueError("To compare questions based on analyst agrement you need to include at least one analyst")
    present_analysts = set(scoresheet_dataframe[analyst_id_column_name])
    for analyst in analysts_to_rate_over:
        if analyst not in present_analysts:
            raise ValueError("Couldn't find any data for analyst "+analyst+". Make sure the input scoresheet dataframe is complete")

    #every analyst in the scoresheet other than the ones specified by the user will be excluded
    analysts_to_exclude = present_analysts - set(analysts_to_rate_over)
    
    #pivot the scoresheet dataset into what is called a "comparison dataframe" format
    question_comparison_dataframe = _create_analyst_comparison_dataframe(scoresheet_dataframe, questions_to_include_in_comparison=questions_to_rate, 
                analyst_id_column_name= analyst_id_column_name, reference_analysts=[], exclude_these_analysts=analysts_to_exclude, subject_id_column_name= subject_id_column_name, indeterminate_answer_code=9, debug=False)

    #compute every supported agreement metric, one at a time
    question_reliability_output = {}
    
    for rating_method in supported_rating_methods:
    
        #call on some helper functions to run comparisons on a question and analyst levels        
        reliabilities_df = get_separate_reliabilities_for_all_analysts_and_questions(question_comparison_dataframe, calculation_to_do=rating_method,
                questions_to_include=questions_to_rate, reference_column_name=THE_OTHER_ANALYSTS, debug=False)
        reliabilities_df = reliabilities_df.sort_values(analyst_id_column_name)
        question_reliabilities = reliabilities_df.groupby('question').apply(rating_aggregation_method)
        question_reliabilities = question_reliabilities.sort_index()
        question_reliability_output[rating_method] = [0.0] * len(questions_to_rate)

        #put the reliability scores into the output data structure
        if rating_method == 'fleiss_kappa':
            for question, reliability in zip(reliabilities_df['question'].values, reliabilities_df['reliability'].values):
                question_reliability_output[rating_method][questions_to_rate.index(question)] = reliability
        else:
            for question, reliability in zip(question_reliabilities.index, question_reliabilities['reliability'].values):
                question_reliability_output[rating_method][questions_to_rate.index(question)] = reliability


    #now let's evaluate the 'coverage' rate per question per analyst
    question_coverage_output = []
    coverage_df = question_comparison_dataframe[[column_name for column_name in question_comparison_dataframe.columns if column_name=='question' or column_name.startswith('determinate_answer_by_analyst_')]].groupby('question').apply(rating_aggregation_method, axis=0).apply(rating_aggregation_method, axis=1)
    for question in questions_to_rate:
        question_overall_coverage = coverage_df[question]
        question_coverage_output.append(question_overall_coverage)
    
    #format the output into DataFrame to return to caller  
    output_dataframe = pd.DataFrame({'question':questions_to_rate, 'coverage':question_coverage_output})  
    for rating_method in supported_rating_methods:
        output_dataframe['agreement_using_'+rating_method] = question_reliability_output[rating_method]
 
    return output_dataframe 

def rate_questions_on_analyst_agreement_with_reference(scoresheet_dataframe, questions_to_rate, analysts_to_rate_over, reference, rating_aggregation_method=np.mean, analyst_id_column_name='analyst', subject_id_column_name='submission'):


    ''' This function is what you should use to determine the analyst inter-agreement on a question by question basis
    
    Functionally, it will perform the rating method on every analyst and for every question. Then it will
    aggregate these ratings across analysts using an aggregation method of your choice.

    inputs:
       scoresheet_dataframe: of raw data, in format of one row per subject and per question, and one column per analyst
       questions_to_rate: list of questions to rate
       analysts_to_rate_over: list of analysts to include in the rating calculations
       rating_aggregation_method: function to combine individual agreement calculations
    '''

    #make sure we're dealing with lists in case the user passed some other iterable
    questions_to_rate = list(questions_to_rate)
    analysts_to_rate_over = list(analysts_to_rate_over)

    supported_rating_methods = ['percent_agreement', 'cohen_kappa'] #weighted cohen kappa, fleiss_kappa, and weighted fleiss kappa currently turned off

    #make sure all analysts the user wants to rate over actually exist in the passed dataframe
    if analysts_to_rate_over is None or len(analysts_to_rate_over)<1:
            raise ValueError("To compare questions based on analyst agrement you need to include at least one analyst")
    present_analysts = set(scoresheet_dataframe[analyst_id_column_name])
    for analyst in analysts_to_rate_over:
        if analyst not in present_analysts:
            raise ValueError("Couldn't find any data for analyst "+analyst+". Make sure the input scoresheet dataframe is complete")


    #make sure the reference analyst exists in the passed dataframe
    if reference is None:
            raise ValueError("To compare analyst to reference you need to specify a valid analyst or instrument as reference")
    if reference not in present_analysts:
            raise ValueError("Couldn't find any data for reference "+reference+". Make sure the input scoresheet dataframe is complete")

    #an analyst cannot be chosen as both a reference and as a rated analyst
    if reference in analysts_to_rate_over:
            raise ValueError("Cannot rate analyst "+reference+" while also using him/her as a reference")
    

    #every analyst in the scoresheet other than the ones specified by the user will be excluded
    analysts_to_exclude = present_analysts - set(analysts_to_rate_over+[reference]) 
    
    #pivot the scoresheet dataset into what is called a "comparison dataframe" format
    question_comparison_dataframe = _create_analyst_comparison_dataframe(scoresheet_dataframe, questions_to_include_in_comparison=questions_to_rate, 
                analyst_id_column_name= analyst_id_column_name, reference_analysts=[reference], exclude_these_analysts=analysts_to_exclude, subject_id_column_name= subject_id_column_name, indeterminate_answer_code=9, debug=False)

    #compute every supported agreement metric, one at a time
    question_reliability_output = {}
    
    for rating_method in supported_rating_methods:

        #call on some helper functions to run comparisons on a question and analyst levels        
        reliabilities_df = get_separate_reliabilities_for_all_analysts_and_questions(question_comparison_dataframe, calculation_to_do=rating_method,
                questions_to_include=questions_to_rate, reference_column_name=reference, debug=False)
        reliabilities_df = reliabilities_df.sort_values(analyst_id_column_name)
        question_reliabilities = reliabilities_df.groupby('question').apply(rating_aggregation_method)
        question_reliabilities = question_reliabilities.sort_index()
        question_reliability_output[rating_method] = [0.0] * len(questions_to_rate)

        #put the reliability scores into the output data structure
        if rating_method == 'fleiss_kappa':
            for question, reliability in zip(reliabilities_df['question'].values, reliabilities_df['reliability'].values):
                question_reliability_output[rating_method][questions_to_rate.index(question)] = reliability
        else:
            for question, reliability in zip(question_reliabilities.index, question_reliabilities['reliability'].values):
                question_reliability_output[rating_method][questions_to_rate.index(question)] = reliability


    #now let's evaluate the 'coverage' rate per question per analyst
    question_coverage_output = []
    coverage_df = question_comparison_dataframe[[column_name for column_name in question_comparison_dataframe.columns if column_name=='question' or column_name.startswith('determinate_answer_by_analyst_')]].groupby('question').apply(rating_aggregation_method, axis=0).apply(rating_aggregation_method, axis=1)
    for question in questions_to_rate:
        question_overall_coverage = coverage_df[question]
        question_coverage_output.append(question_overall_coverage)
    
    #format the output into DataFrame to return to caller        
    output_dataframe = pd.DataFrame({'question':questions_to_rate, 'coverage':question_coverage_output})  
    for rating_method in supported_rating_methods:
        output_dataframe['agreement_with_'+reference+'_using_'+rating_method] = question_reliability_output[rating_method]
 
    return output_dataframe 


#########
### 
###   HELPER FUNCTIONS
###
#########



def _create_analyst_comparison_dataframe(analyst_scoresheet_dataframe, questions_to_include_in_comparison, analyst_id_column_name, reference_analysts, exclude_these_analysts, subject_id_column_name, other_groupby_keys=[], prior_analyst_weights_hypothesis_dict=None, indeterminate_answer_code=9, debug=False):
    ''' This is the main API endpoint to convert an input dataframe into a format convenient for analysis by the other API in this lib 
    
        An analyst_comparison_dataframe contains one unique (submission, question) per row, and multiple columns, one per analyst answer
        
        
        analyst_scoresheet_dataframe: input dataframe, expected to contain one unique (submission, analyst) per row, and multiple columns, one per question, containing question answers in the proper enconding

        reference_analysts: this is a list of analyst names that should not be used in the analysis (presumably because they will be
		      considered as ground truth options to compared with). These will not have the analyst naming convention enforced on them.


		This function also enforces a naming convention that every analyst column must have a prefix 'analyst_'.
		The analysis code looks for this prefix to decide which columns are for the analysts that should be evaluated.
		
		prior_analyst_weights_hypothesis_dict: relative weights per analyst. These are used when combining analyst judgments using majority vote
		
    '''
    
    def _enforce_analyst_naming_convention(name):
        ''' needed so that after conversion to analysts into columns it is clear which columns are for real analysts 
        '''
        if name in reference_analysts:
            #### This is a special case 
            return name
        name = 'analyst_'+str(name) if 'analyst_' not in str(name) else name
        return name


    def _get_combined_df(in_df, group_keys, questions_to_include_in_comparison, analyst_id_column_name, weights_key=None, new_name=THE_OTHER_ANALYSTS, exclude_analyst_list=[]):
        ''' Helper function for evaluate_analyst_reliabilities_and_pickle_combination_function
        does an initial combination of the input dataframe before running the reliability analysis
        Inputs:
            in_df: the dataframe on which to run the combination
            group_keys: perform the combination in pieces on these groups
            analyst_id_column_name: the column that defines the analysts that are aggregated together
            weights_key: if you have a prior weights assumption, pass it here
            exclude_analyst_list: if you want combination to not include some analysts. Common
               if you are trying to compare one against others
        Return:
            a new dataframe that contains only the new rows tagged on with combined values
        '''
        ### Make sure to reset indices at beginning to ensure things are not scrambled later
        df = in_df.reset_index()
        if exclude_analyst_list != []:
            df = df[~df[analyst_id_column_name].isin(exclude_analyst_list)]
        #print 'after exclusion, df: ', df
        if weights_key is None:   ### Treat all rows equally
            weights_key = 'prior_analyst_weights'
            df[weights_key] = [1.]*len(df.index)
        weighted_mode_operation = lambda x: weighted_mode(x, weights=df.loc[x.index, weights_key])
        def weighted_mode(grouped_series, weights):
            #### Maybe replace this by a call to weighted_mode_combination once you have other pieces working??
            values_to_agg = grouped_series.values
            unique_values = np.unique(values_to_agg)

            value_weight_dict = collections.OrderedDict()   ### in case of ties want consistent ordered results
            max_value = None
            max_weight = None
            for value in unique_values:
                these_weights = weights[values_to_agg==value]
                total_weight = np.sum(weights[values_to_agg==value])
                value_weight_dict[value] = total_weight

                if max_weight is None or max_weight < total_weight:
                    max_value = value
                    max_weight = total_weight
            return max_value

        ### Do weighted mode calculation on questions, and just take the first observed value in any non-questions:
        grouped_df = df.groupby(group_keys)
        agg_functions_to_apply = {column_name: weighted_mode_operation for column_name in questions_to_include_in_comparison}
        spectator_column_names = [column_name for column_name in df.columns if column_name not in questions_to_include_in_comparison]
        for spectator_column_name in spectator_column_names:
            agg_functions_to_apply[spectator_column_name] = lambda x: x.values[0]
        out_feature_df = grouped_df.agg(agg_functions_to_apply)
        out_feature_df[analyst_id_column_name] = [new_name]*len(out_feature_df.index)
        return out_feature_df

    def _build_all_permutation_of_exclusions_combined_df(analysis_df, group_keys, questions_to_include_in_comparison, analyst_id_column_name, weights_key, all_analyst_names):
        ''' Build all permutations of _get_combined_df with one analyst held out. This is the format that will be needed for downstream analysis. '''
        reliability_analysis_df = None
        for analyst in all_analyst_names:
            this_comb_name = ALL_ANALYSTS_EXCEPT +str(analyst)
            this_analyst_combined_df = _get_combined_df(analysis_df, group_keys=group_keys, questions_to_include_in_comparison=questions_to_include_in_comparison,
                analyst_id_column_name=analyst_id_column_name, weights_key=weights_key, new_name=this_comb_name, exclude_analyst_list=[analyst])
            if reliability_analysis_df is None: reliability_analysis_df = pd.concat([analysis_df, this_analyst_combined_df], ignore_index=True)
            else: reliability_analysis_df = pd.concat([reliability_analysis_df, this_analyst_combined_df], ignore_index=True)
        return reliability_analysis_df


    if exclude_these_analysts == []:
       analysis_df = cp.deepcopy(analyst_scoresheet_dataframe)
    else:
       analysis_df = analyst_scoresheet_dataframe[~analyst_scoresheet_dataframe[analyst_id_column_name].isin(exclude_these_analysts)]


    analysis_df[analyst_id_column_name] = analysis_df[analyst_id_column_name].apply(_enforce_analyst_naming_convention)
    all_analyst_names = np.array([name for name in np.unique(analysis_df[analyst_id_column_name].values) if name.startswith('analyst_')])

    truth_output_column_names = reference_analysts

    group_keys = [subject_id_column_name] + other_groupby_keys
    if prior_analyst_weights_hypothesis_dict is not None:
        analysis_df['analyst_weight'] = full_df[analyst_id_column_name].apply(lambda x: prior_analyst_weights_hypothesis_dict[x])
        analysis_df['prior_analyst_weights'] = [prior_analyst_weights_hypothesis_dict[aid] for aid in analysis_df[analyst_id_column_name].values]
        prior_weights_key = 'prior_analyst_weights'
    else: prior_weights_key = None
    reliability_analysis_df = _build_all_permutation_of_exclusions_combined_df(analysis_df, group_keys=group_keys, questions_to_include_in_comparison=questions_to_include_in_comparison,
        analyst_id_column_name=analyst_id_column_name, weights_key=prior_weights_key, all_analyst_names=all_analyst_names)


    #pivot dataframe to key on "comparison format"
    if debug:
        print 'Now convert data frame to format for reliability analysis'
    analyst_comparison_dataframe = _convert_df_from_questions_in_columns_to_rows(reliability_analysis_df, questions_to_include=questions_to_include_in_comparison, subject_id_column_name=subject_id_column_name,
              analyst_id_column_name=analyst_id_column_name, other_groupby_keys=other_groupby_keys)
    
    
    #add determinate answer info as boolean columns, one per analyst
    for analyst in all_analyst_names:
        analyst_comparison_dataframe['determinate_answer_by_'+analyst] = np.where(analyst_comparison_dataframe[analyst]==indeterminate_answer_code, 0, 1)

    
    return analyst_comparison_dataframe 













def get_separate_reliabilities_for_all_analysts_and_questions(dataframe, calculation_to_do, questions_to_include, reference_column_name=None, debug=True, **kwargs):
    ''' Get reliability metrics for one or more questions for all contained analysts.
    This function is like get_reliabilities_for_given_question, except that there is an 
    additional column in the input to specify the question being asked. These questions are
    looped over and get_reliabilities_for_given_question is called for each question

    inputs:
        dataframe: should be encoded in desired format before passing to this function. There should
                   be one column for each analyst WHICH MUST USE PREFIX 'analyst_', and one row for
                   each subject, with entries representing the responses. Another column should represent
                   the question that the response is for. The func
        calculation_to_do: reliability calculation to perform
        questions_to_include: list of questions to calculate reliabilities on
        reference_column_name: "ground truth" column to calculate reliabilities with respect to (might be a combination of other analysts
    '''
    each_question_reliabilities_dfs = []
    for question in questions_to_include:
        this_question_df = dataframe[dataframe['question']==question]
        if len(this_question_df.index)==0:
            print 'Question ', question, ' not present in data. Skip it.'
            continue
        this_question_reliabilities_df = _get_reliabilities_for_given_question(this_question_df, calculation_to_do,
                question, reference_column_name, **kwargs)
        each_question_reliabilities_dfs.append(this_question_reliabilities_df)
    aggregate_reliabilities_df = pd.concat(each_question_reliabilities_dfs, ignore_index=True)
    return aggregate_reliabilities_df


def _get_reliabilities_for_given_question(dataframe, calculation_to_do, question, reference_column_name=None, **kwargs):
    ''' Get reliability metrics for some number of analysts and evaluations on a particular question.
    inputs:
        dataframe: should be encoded in desired format before passing to this function. There should
                   be one column for each analyst WHICH MUST USE PREFIX 'analyst_', and one row for
                   each subject, with entries representing the responses. Should also have a reference column
                   if desired as a ground truth to evaluate analysts against.
                   Note: Missing/not answered results should be included as either np.nan, 'nan', or ''. 
                       All other responses are assumed to be valid.
        calculation_to_do: string to specify which reliability metric to calculate. Suggested options to support:
			==> To get one reliability metric per analzer. Options:
                 ** percent agreement
                 ** cohen_kappa (improvement on percent agreement)
                 ** weighted_cohen_kappa (as cohen_kappa but penalty for disagreement depends on type of disagreement).
                 for example, you might want a '2' vs '3' disagreement to be less severe than a '0' vs '3' disagreement.
                 In this case a dictionary of weight definitions needs to be specified in **kwargs
            ==> To get one reliability metric over all analysts (reference_column must be None to use this):
                 ** fleiss kappa, or weighted fleiss kappa (also requires weights specified in **kwargs)
        reference_column_name: name of column that reliability is compared with (not defined for fleiss kappas)
        question: the question being evaluated
    returns:
        If calculation_to_do specifies that there should be a metric for each analyst, returns a dictionary that 
        contains one reliability value per analyst. If calculation_to_do specifies a single metric over all analysts
        (such as the fleiss kappa), returns a single float as reliability.
    '''
    if 'question' in dataframe.columns:
        ## Make sure he same question is being consistently evaluated
        all_questions = np.unique(dataframe['question'].values)
        if len(all_questions)>1:
            raise ValueError('Should only pass data about a single question to get_reliabilities_for_given_question. '+\
                    ' The following questions were observed: '+str(all_questions))


    analysts_to_evaluate = [column_name for column_name in dataframe.columns if column_name.startswith('analyst_')]
    if len(analysts_to_evaluate)==0:
        raise ValueError('No analysts found in columns of input dataframe. Did you forget to specify "analyst_"'+\
                ' as prefix?')

    if calculation_to_do in ['percent_agreement', 'cohen_kappa', 'weighted_cohen_kappa']:
        separate_reliability_estimate_for_each_analyst = True
    elif calculation_to_do in ['fleiss_kappa', 'weighted_fleiss_cappa']:
        separate_reliability_estimate_for_each_analyst = False
    else:
        raise ValueError('calculation_to_do '+calculation_to_do+' not understood')
    reliability_results = {'question': [], 'reference': [], 'analyst': [], 'calculation': [], 'reliability': []}
    if separate_reliability_estimate_for_each_analyst:
        for analyst in analysts_to_evaluate:
            if reference_column_name == THE_OTHER_ANALYSTS:
                this_reference_column_name = ALL_ANALYSTS_EXCEPT +str(analyst)
            else:
                this_reference_column_name = reference_column_name
            
            
            if analyst not in dataframe:
            	raise ValueError('_get_reliabilities_for_given_question(): dataframe should contain analyst column '+analyst+" but doesn't")
            if this_reference_column_name not in dataframe:
             	raise ValueError('_get_reliabilities_for_given_question(): dataframe should contain this_reference_column_name column '+ str(this_reference_column_name) +" but doesn't")
             	
             	
            #HALIM: IF THE separate_reliability_estimate_for_each_analyst is TRUE, AND reference IS NOT THE_OTHER_ANALYSTS ,
            #THEN FILTER dataframe TO EXCLUDE ALL INSTANCES WHERE THE analyst IN QUESTION GAVE AN INDETERMINATE ANSWER
            if (reference_column_name!=THE_OTHER_ANALYSTS and separate_reliability_estimate_for_each_analyst):
                filtered_dataframe = dataframe[dataframe["determinate_answer_by_"+analyst]==1] 
            else:
                filtered_dataframe = dataframe.copy(deep=True)
                
            reliability = _get_reliability_for_given_analyst_and_question(filtered_dataframe[analyst].values, filtered_dataframe[this_reference_column_name].values,
                calculation_to_do, **kwargs)
            reliability_results['analyst'].append(analyst)
            reliability_results['reliability'].append(reliability)
            reliability_results['reference'].append(reference_column_name)
            reliability_results['question'].append(question)
            reliability_results['calculation'].append(calculation_to_do)
    else:
        #raise NotImplementedError('calculation_to_do '+calculation_to_do+' not implemented yet')
        reliability = _get_reliability_given_all_analysts_for_question(dataframe, analysts_to_evaluate,
                calculation_to_do, **kwargs)
        reliability_results['analyst'].append( 'all_analysts_aggregate')
        reliability_results['reliability'].append( reliability)
        reliability_results['reference'].append(reference_column_name)
        reliability_results['question'].append(question)
        reliability_results['calculation'].append(calculation_to_do)
    reliability_results_df = pd.DataFrame(reliability_results)
    return reliability_results_df

def _is_this_value_missing(element, missing_values_to_check=['', np.nan, 'nan']):
    ''' Helper function to check a given element to see if it is
    of a type we would consider to be "missing" '''
    if element is None:
        return True
    for missing_value in missing_values_to_check:
        ### Checking nans in a way that doesn't crash on strings is complicated.
        ### There must be some better way to do this???
        try:
            if np.isnan(missing_value):
                if np.isnan(element): return True   ### Equality check invalid
        except:
            pass ### not nan (what's a better way to do this safely?)
        if element == missing_value: return True
    return False

def _no_missing_values_in_row(x, analysts_to_evaluate):
    for analyst in analysts_to_evaluate:
        if _is_this_value_missing(x[analyst]): return False
    return True

def _get_reliability_given_all_analysts_for_question(dataframe, analysts_to_evaluate, calculation_to_do, **kwargs):
    ''' Calculate reliability for a all analysts for a given question. 
    Inputs:
        dataframe: results for all analysts for a given question. Some values may be emptyif not all completed.
        calulation_to_do: only "fleiss_kappa" implemented for this approach at the moment
    Returns:
        float of reliability result
    Note: Missing/not answered results should be included as either np.nan, 'nan', or ''. 
        All other responses are assumed to be valid.
    '''

    def reformat_df_for_fleiss_kappa(in_df):
        ''' Assumes input is DF with analysts in columns, subjects in rows, and each response
        in the field entries. Want to reformat to dataframe with subjects in rows, response categories in 
        columns, and counts of number of analysts responding with that rating for that subject in each entry '''

        ### First transpose to get subjects in columns and analysts in rows:
        #print 'in_df: ', in_df
        transpose_df = in_df.transpose()
        ### Now count responses for each subject:
        count_dict = {}
        for column_name in transpose_df.columns:
            count_dict[column_name] = transpose_df[column_name].value_counts()
        count_df = pd.DataFrame(count_dict)
        ### Now have subjects in columns and response categories in rows with counts in entries. Re-transpose.
        result_df = count_df.transpose()
        ### Nan's where values are missing. Replace with zeros.
        result_df = result_df.fillna(0)
        ### Make sure to recover original indexing if any
        result_df.index = in_df.index

        return result_df
        
        
    if calculation_to_do not in ['fleiss_kappa']:
        raise ValueError('calculation_to_do '+calculation_to_do+' not defined for all analysts calculation')
    all_completed_df = dataframe[dataframe.apply(_no_missing_values_in_row, axis=1, args=(analysts_to_evaluate,))]
    if calculation_to_do == 'fleiss_kappa':
        refornatted_for_fleiss_df = reformat_df_for_fleiss_kappa(all_completed_df[analysts_to_evaluate])
        
        print refornatted_for_fleiss_df

        fleiss_kappa = _compute_fleiss_kappa(refornatted_for_fleiss_df.values.tolist())
        return fleiss_kappa


def _get_reliability_for_given_analyst_and_question(analyst_results, reference_results, calculation_to_do, **kwargs):
    ''' Calculate reliability for a particular analyst. 
    Inputs:
        analyst_results: a numpy array of results for one or more analysts. Some values may be emptyif not all completed.
        reference_results: the "ground truth" reference numpy array of results. Some values may be null if not all completed.
        calulation_to_do: see documentation of get_reliabilities_for_given_question
    Returns:
        float of reliability result
    Note: Missing/not answered results should be included as either np.nan, 'nan', or ''. 
        All other responses are assumed to be valid.
    '''

    def get_results_completed_by_both(analyst_results, reference_results):
        ''' analyst results and reference results may be missing data. Return arrays that contain only matching values for both arrays '''
        if len(analyst_results) != len(reference_results):
            raise ValueError('Mismatch between analyst_results and reference_result array length')
        ok_analyst_results = []
        ok_reference_results = []
        for idx in range(len(analyst_results)):
            if _is_this_value_missing(analyst_results[idx]) or _is_this_value_missing(reference_results[idx]):
                continue
            else:
                ok_analyst_results.append(analyst_results[idx])
                ok_reference_results.append(reference_results[idx])
        return np.array(ok_analyst_results), np.array(ok_reference_results)

    
    if calculation_to_do not in ['percent_agreement', 'cohen_kappa', 'weighted_cohen_kappa']:
        raise ValueError('calculation_to_do '+calculation_to_do+' not defined for analyst vs reference calculation')
    valid_analyst_results, valid_reference_results = get_results_completed_by_both(analyst_results, reference_results)
    if calculation_to_do == 'percent_agreement':
        n_results_tot = len(valid_analyst_results)
        n_results_agreeing = len([analyst_value for analyst_value, reference_value in\
            zip(analyst_results, reference_results) if analyst_value == reference_value])
        frac_agreement = np.nan if n_results_tot==0 else float(n_results_agreeing) / float(n_results_tot)
        return frac_agreement

    if calculation_to_do == 'cohen_kappa':
        if len(valid_analyst_results) == 0 or len(valid_reference_results) == 0:
            print 'Not enough common valid results. Abort with value -999 for cohen kappa.'
            return -999
        kappa_value = metrics.cohen_kappa_score(valid_analyst_results, valid_reference_results)
        return kappa_value

    if calculation_to_do == 'weighted_cohen_kappa':
                            
        weights = kwargs['weights'] if 'weights' in kwargs else [1]*len(valid_analyst_results)
        ### sklearn only has weights supported here in version 0.18 and later
        try:
            kappa_value = metrics.cohen_kappa_score(valid_analyst_results, valid_reference_results, weights=weights)
        except Exception as exception_msg:
                print 'Failed to evaluate cohen kappa with weights. Are you using an older version of sklearn (only implemented in version 0.18 and later)?'
                raise ValueError(exception_msg)
        return kappa_value
        



def _compute_fleiss_kappa(mat, debug=False):
    """ Copied  with minor reformatting from https://en.wikibooks.org/wiki/Algorithm_Implementation/Statistics/Fleiss%27_kappa

        Computes the Kappa value
        @param n Number of rating per subjects (number of human raters)
        @param mat Matrix[subjects][categories]
        @return The Kappa value """

    def checkEachLineCount(mat):
        """ Assert that each line has a constant number of ratings
            @param mat The matrix checked
            @return The number of ratings
            @throws AssertionError If lines contain different number of ratings """
        n = sum(mat[0])
    
        assert all(sum(line) == n for line in mat[1:]), "Line count != %d (n value)." % n
        return n

    n = checkEachLineCount(mat)   # PRE : every line count must be equal to n
    N = len(mat)
    k = len(mat[0])
    
    if debug:
        print n, "raters."
        print N, "subjects."
        print k, "categories."
    
    # Computing p[]
    p = [0.0] * k
    for j in xrange(k):
        p[j] = 0.0
        for i in xrange(N):
            p[j] += mat[i][j]
        p[j] /= N*n
    if debug: print "p =", p
    
    # Computing P[]    
    P = [0.0] * N
    for i in xrange(N):
        P[i] = 0.0
        for j in xrange(k):
            P[i] += mat[i][j] * mat[i][j]
        P[i] = (P[i] - n) / (n * (n - 1))
    if debug: print "P =", P
    
    # Computing Pbar
    Pbar = sum(P) / N
    if debug: print "Pbar =", Pbar
    
    # Computing PbarE
    PbarE = 0.0
    for pj in p:
        PbarE += pj * pj
    if debug: print "PbarE =", PbarE
    
    kappa = (Pbar - PbarE) / (1 - PbarE)
    if debug: print "kappa =", kappa
    
    return kappa

### Helper function if needed depending on the format of the input dataframe for the reliability analysis
def _convert_df_from_questions_in_columns_to_rows(in_df, questions_to_include, subject_id_column_name, analyst_id_column_name, spectator_column_names=[], other_groupby_keys=[]):
    ''' Assumes input has format with a single column for subject Id, a single column for analyst id, and
    a different column for each question. 

    Outputs dataframe in the format the rest of this code expects: one column per analyst, and each row being
    a set of responses for a particular subject and a particular question. 

    inputs:
        in_df: pre-converted dataframe. 
           An example format might be:
             Cognoa Id, analyst Id, ados1_a1, ados1_a3, ...
               10          121          0          2 ...
               11          121          0          1 ...
               ...
        questions_to_include: these are the columns that should be considered questions, each of which will become a different row
        spectator_column_names: columns that do not fall into any other category, but which you want to keep in the output dataframe
        other_groupby_keys: when joining output across analysts, what additional matching criteria is there besides subject id and
              question (if any). Common example: triton video version.

    returns:
        converted format dataframe
           An example format might be:
             Cognoa Id, question, analyst_121, analyst_192, ...
               10        ados1_a1         0       2 ...
               10        ados1_a3         2       1 ...
               ...
    
    '''

    unique_analysts = np.unique(in_df[analyst_id_column_name].values)
    result_df = None
    for analyst in unique_analysts:

        analyst_str = str(analyst)
        this_analyst_df = in_df[in_df[analyst_id_column_name]==analyst]

        ### The analyst_str part of this dictionary will be filled with responses
        results_for_this_analyst = {analyst_str: [], subject_id_column_name: [], 'question': []}
        for column_name in spectator_column_names+other_groupby_keys:
            results_for_this_analyst[column_name] = []
        ### this df has columns of questions and rows of subjects. Want to collapse to single
        ### for answer with the question name in new column
        for question in questions_to_include:
            subject_responses = this_analyst_df[question].values
            subject_ids = this_analyst_df[subject_id_column_name].values
            questions = [question]*len(subject_ids)

            results_for_this_analyst[analyst_str] += list(subject_responses)
            results_for_this_analyst[subject_id_column_name] += list(subject_ids)
            results_for_this_analyst['question'] += questions
            for column_name in spectator_column_names+other_groupby_keys:
                results_for_this_analyst[column_name] += list(this_analyst_df[column_name].values)

        this_analyst_out_df = pd.DataFrame(results_for_this_analyst)
        if result_df is None:
            result_df = cp.deepcopy(this_analyst_out_df)
        else:
            questions_to_merge_on = [subject_id_column_name, 'question']+other_groupby_keys
            if 'Triton Video Version' in result_df.columns:
                questions_to_merge_on.append('Triton Video Version')
            result_df = result_df.merge(this_analyst_out_df, on=questions_to_merge_on, how='outer')

    return result_df


#########
### 
###   API TO GENERATE ANALYST RATIING COMBINATION FUNCTIONS
###
#########


def evaluate_analyst_reliabilities_and_create_combination_function(in_df, out_filename, reliability_calculation_to_do,
        questions_to_include, reliability_combination_function=np.mean, reference_analyst=THE_OTHER_ANALYSTS, combination_function=None,
        use_combination_for_reliability_analysis=True, subject_id_column_name='Clinical Study Id', analyst_id_column_name='Analyst Id', exclude_these_analysts=[],
        other_groupby_keys=[], prior_analyst_weights_hypothesis_dict=None, how_serialized='pickle', debug=False):
    ''' Runs full reliability calculation, and outputs pickled combination function.
    Inputs:
        in_df: an analysis dataframe in a format with one row per combination of subject and analyst,
               and columns representing different question responses. If encoding is desired it must
               be done before calling this function.
        out_filename: purpose of this function is to create this pickled file
        reliability_calculation_to_do: Run this reliability calculation on each question for each analyst
			==> To get one reliability metric per analzer. Options:
                 ** percent agreement
                 ** cohen_kappa (improvement on percent agreement)
                 ** weighted_cohen_kappa (as cohen_kappa but penalty for disagreement depends on type of disagreement).
                 for example, you might want a '2' vs '3' disagreement to be less severe than a '0' vs '3' disagreement.
                 In this case a dictionary of weight definitions needs to be specified in **kwargs
            ==> To get one reliability metric over all analysts (reference_column must be None to use this):
                 ** fleiss kappa, or weighted fleiss kappa (also requires weights specified in **kwargs)
        questions_to_include: list of questions that should be considered in reliability calculation
        reliability_combination_function: Run this function to combine reliabilities across functions for each analyst
        reference_analyst: If you want your reliability to be with respect to a ground truth analyst (such
               as the thompson ADOS results), specify that column here. In that case use_combination_for_reliability_analysis
               should be set to True. Otherwise use_combination_for_reliability_analysis
               should be False, and reference_analyst should be "combined" to use the results of this combination.
        combination_function: The function to use when performing a combination
        use_combination_for_reliability_analysis: If you will run a combination and then use it in the reliability analysis, make this flag true

        .... Should be True if reference_analyst set to "combined", otherwise False
        other parameters only matter if this is true:
            subject_id_column_name: optional if you want to redfine the column of which subjects are which across analysts
            analyst_id_column_name: optional if you want to redefine the column of which analysts are which
            other_groupby_keys: optional, if you want to group by more than just the subjects when doing the initial combination
               (recommended to also group by the "Triton Video Version" since analysts sometimes are inconsistent here)
            prior_analyst_weights_hypothesis_dict: if you want initial reliability evaluation to use a combination with a prior weight hypothesis
               specify a dict with analyst Ids as keys and weights as values.

    Important note on missing values:
         Real missing data should be entered as either np.nan, 'nan', or ''. If they are left as some other value such as 0 or 8 then module
         will consider this to be a real category to compare for accruacy. For example, if you have do a combined analysis across module 1 and module 2
         versions, and each subject has only been evaluated with one or the other, then make sure responses to the non-evaluated questions are
         one of these three values.
    Returns:
        Dictionary that contains the function and associated kwargs that gets serialized (in case user wants to use it immediately).
    '''



    def _weighted_mode_combination(analyst_scores_dict, analyst_weights_dict):
        ''' Given a single dictionary of analyst responses and the reliability weights of each analyst,
        do a weighted mode combination. '''
        for weight in analyst_weights_dict.values():
            if not isinstance(weight, numbers.Number):
                raise TypeError('Error, weighting '+str(weight)+' not numeric')
            if not np.isfinite(weight):
                raise ValueError('Error, weighting '+str(weight)+' invalid')
        value_weight_dict = {}
        for analyst, score in analyst_scores_dict.iteritems():
            if analyst not in analyst_weights_dict:
                raise KeyError('analyst '+str(analyst)+' not recognized')
            if score not in value_weight_dict.keys():
                value_weight_dict[score] = analyst_weights_dict[analyst]
            else:
                #### Negative weights can arise from metrics like
                #### Kappa if we have analysts who perform worse than
                #### Random. These analysts should not be counted, rather
                #### than counted as an anti-vote.
                if analyst_weights_dict[analyst]>0:
                    value_weight_dict[score] += analyst_weights_dict[analyst]
        if len(value_weight_dict.values())==0:
            raise ValueError('No valid analysts found to do a weighted mode combination')
        weight_list = value_weight_dict.values()
        score_list = value_weight_dict.keys()
        max_weight = max(weight_list)
        max_weight_index = weight_list.index(max_weight)
        score_weighted_mode = score_list[max_weight_index]
        diagnostic_info = {'max_weight': max_weight}
        return score_weighted_mode, diagnostic_info

    if combination_function is None:
        combination_function = _weighted_mode_combination

    if debug:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Run reliability analysis and create pickled file for the following settings:'
        print 'analysts to evaluate: ', np.unique(analysis_df[analyst_id_column_name].values)
        print 'Ground truth: ', reference_analyst
        print 'Reliability calculation to perform: ', reliability_calculation_to_do
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'


    ### First convert input dataframe to the format that is needed for analysis
    analyst_comparison_dataframe = _create_analyst_comparison_dataframe(in_df, questions_to_include=questions_to_include,
	    analyst_id_column_name=analyst_id_column_name, reference_analysts=[reference_analyst], exclude_these_analysts=exclude_these_analysts,
	    use_combination_for_reliability_analysis=use_combination_for_reliability_analysis, subject_id_column_name=subject_id_column_name, 
		other_groupby_keys=other_groupby_keys, prior_analyst_weights_hypothesis_dict=prior_analyst_weights_hypothesis_dict, debug=debug )

    print 'Now evaluate reliabilties'
    analyst_reliabilities = rate_analysts_against_reference(analyst_comparison_dataframe, rating_method=reliability_calculation_to_do,
              questions_to_include=questions_to_include, reference_column_name=reference_analyst, subject_id_column_name=subject_id_column_name,
			  debug=debug)
    print 'Now save picked file'
    if how_serialized == 'pickle':
        combined_function_and_kwargs = _make_pickled_combination_function(combination_function, out_filename=out_filename,
                analyst_weights_dict=analyst_reliabilities, debug=debug)
    else:
        raise NotImplementedError('Have not implemented function serialization scheme: '+how_serialized)
    return combined_function_and_kwargs





def get_combination_function(in_filename, how='pickle'):
    if how=='pickle':
        return get_pickled_combination_function(in_filename)
    else:
        raise NotImplementedError('Serialization scheme '+how+' not implemented yet')

def get_pickled_combination_function(in_filename):
    ''' extract the combination function and any associated arguments from in_filename.
    It is expected that users will call this function when they want to extract the prebuilt
    pickled file. '''
    loaded_params = dill.load(open(in_filename, 'rb'))
    combination_function = loaded_params['function']
    kwargs = {key: value for key, value in loaded_params.iteritems() if key != 'function'}
    return combination_function, kwargs




def _make_pickled_combination_function(combination_function, out_filename, debug=False, **kwargs):
    ''' make a pickled file that contains all necessary information to perform a combination.
    combination_function can refer to any function that performs a combination
    Any necessary arguments to successfully execute combination_function should be passed in the **kwargs.
    They will be pickled as well 

    Returns: pickled object in case user wants to use it immediately
    '''
    pickle_this = {'function': combination_function}
    for key in kwargs.keys():
        pickle_this[key] = kwargs[key]
    if debug:
        print 'pickle_this: ', pickle_this
        print 'out filename: ', out_filename
    dill.dump(pickle_this, open(out_filename, 'wb'))
    return pickle_this

        














