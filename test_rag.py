from query_data import query_rag
from langchain_community.llms.ollama import Ollama

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""


# def test_monopoly_rules():
#     assert query_and_validate(
#         question="How much total money does a player start with in Monopoly? (Answer with the number only)",
#         expected_response="$1500",
#     )
#
#
# def test_ticket_to_ride_rules():
#     assert query_and_validate(
#         question="How many points does the longest continuous train get in Ticket to Ride? (Answer with the number only)",
#         expected_response="10 points",
#     )
# def test_turbopump_question():
#     assert query_and_validate(
#         question="What is the axial inlet portion of the turbopump rotor whose function is to raise the inlet head by an amount sufficient to preclude cavitation?",
#         expected_response="inducer",
#     )
def test_general_test_philosophy():
    assert query_and_validate(
        question="What is the general test philosophy mentioned in the SMC-S-025 standard for liquid rocket engines?",
        expected_response="The general test philosophy emphasizes a verification approach that includes the use of engine samples, modeling, and simulation to validate engine designs and ensure that all functional objectives are met.",
    )

def test_uniform_success_targets():
    assert query_and_validate(
        question="Why are uniform success targets important in engine development testing according to the AeroCorp report?",
        expected_response="Uniform success targets provide consistent performance benchmarks across the industry, ensuring that engine development programs meet standardized expectations and reduce the likelihood of catastrophic failure.",
    )

def test_penalty_for_unlawful_export():
    assert query_and_validate(
        question="What is the penalty for unlawful export of ITAR-controlled technical data according to JANNAF guidelines?",
        expected_response="The penalty can be up to 10 years imprisonment, a fine of $1,000,000, or both.",
    )

def test_throttling_challenge():
    assert query_and_validate(
        question="What is the primary challenge when throttling liquid rocket engines as identified in the study of LRE throttling capabilities?",
        expected_response="The primary challenge is maintaining an adequate pressure drop across the injector to ensure proper atomization and mixing of the propellants, especially at low thrust levels.",
    )

def test_pump_fed_cycle_variables():
    assert query_and_validate(
        question="What are the two main configuration variables that define pump-fed liquid rocket cycles?",
        expected_response="The two main configuration variables are the energy source for the turbine drive and the turbine discharge location.",
    )

def test_nasa_turbopump_monograph():
    assert query_and_validate(
        question="What is the purpose of the NASA monograph on liquid rocket engine turbopump inducers?",
        expected_response="The purpose is to organize and present the significant experience and knowledge accumulated from various development and operational programs to provide design guidance for increased consistency, reliability, and efficiency in turbopump design.",
    )

def test_propellant_feed_system_types():
    assert query_and_validate(
        question="What are the two main types of propellant feed systems discussed in the document, and which one is used for high-pressure applications?",
        expected_response="The two main types are pressure-fed and pump-fed systems. The pump-fed system is used for high-pressure, high-performance applications.",
    )

def test_pressurization_system_function():
    assert query_and_validate(
        question="What is the main function of a pressurization system in a liquid rocket engine?",
        expected_response="The main function is to ensure that the propellants are delivered to the combustion chamber at the required pressure and flow rates, which is critical for engine performance.",
    )

def test_preliminary_design_goal():
    assert query_and_validate(
        question="What was the primary goal of the project discussed in the Preliminary Design and Simulation of LRE?",
        expected_response="The primary goal was to develop the foundations for designing and simulating liquid rocket engines, allowing for preliminary design and control simulations based on different configurations.",
    )

def test_nasa_sp_8107_design_criteria():
    assert query_and_validate(
        question="What is the significance of the design criteria presented in NASA SP-8107 for turbopump systems?",
        expected_response="The design criteria are intended to ensure the successful design of turbopump systems, offering guidelines that increase reliability, reduce costs, and improve efficiency in rocket engine development.",
    )


def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = Ollama(model="mistral")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )
