###########################################################################################
                                        ## license ##
###########################################################################################
'''
MIT License

Copyright (c) 2022 pafloxy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''

###########################################################################################
                                        ## imports ##
###########################################################################################



import numpy as np
from numpy import pi
import math
import seaborn as sns
from IPython.display import Image
import matplotlib.pyplot as plt
from typing import Union, Iterable, Optional

from qiskit import *
from qiskit.circuit.library import *
from qiskit.algorithms import *
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit #, InstructionSet
from qiskit import quantum_info, IBMQ, Aer
from qiskit.quantum_info import partial_trace, Statevector, state_fidelity
from qiskit.utils import QuantumInstance
from qiskit.extensions import HamiltonianGate
from qiskit.circuit.quantumregister import Qubit
from qiskit.visualization import plot_histogram, plot_state_qsphere, plot_bloch_multivector, plot_bloch_vector

qsm = Aer.get_backend('qasm_simulator')
stv = Aer.get_backend('statevector_simulator')
aer = Aer.get_backend('aer_simulator')


##################################################################################################
                                    ## helper functions ##
##################################################################################################


def measure_and_plot(qc: QuantumCircuit, shots:int= 1024, show_counts:bool= False, return_counts:bool= False, measure_ancilla: bool = False, decimal_count_keys:bool = True , ancilla_specifier:Union[int, list, str] = 'all'):
    """ Measure and plot the state of the data registers, optionally measure the control ancillas, without modifying the original circuit.
        
        ARGS:
        ----
            qc : 'QuantumCircuit' 
                 the circuit to be measured

            shots: 'int' 
                    no. of shots for the measurement

            show_counts : 'bool' 
                           print the counts dictionary

            measure_ancilla : 'bool' 
                             indicates whether to measure the control ancilla registers.
            
            return_counts : 'bool'
                            returns the counts obtained if True, else retruns the histogram plot

            measure_ancilla: 'bool'
                             indicates whether to measure the controll ancill qubits
                
            decimal_count_keys: 'bool'
                                if 'True' converts the binary state of the controll ancilllas to integer represntation

            ancilla_specifier : 'int' 
                                inidicates whihch of the control registers to meausure, 
                                for eg. ancilla_specifier= 1 refers to the first control ancilla
                                ancilla_specifier= 'all' refers to all the ancillas                                                           
        RETURNS:
        -------
            plots histogram over the computational basis states

     """
    qc_m = qc.copy()
    creg = ClassicalRegister( len(qc_m.qregs[0]) )
    qc_m.add_register(creg)
    qc_m.measure(qc_m.qregs[0], creg)

    if measure_ancilla== True:

        if isinstance(ancilla_specifier, int):
            print('int')##cflag
            if ancilla_specifier > len(qc_m.qregs) or ancilla_specifier < 1: raise ValueError(" 'ancilla_specifier' should be less than no. of control registers and greater than 0")

            creg_cntrl = ClassicalRegister(len(qc_m.qregs[ancilla_specifier]))
            qc_m.add_register(creg_cntrl)
            qc_m.measure(qc_m.qregs[ancilla_specifier], creg_cntrl )

        elif isinstance(ancilla_specifier, list ):
            print('list')##cflag
            for ancilla in ancilla_specifier:

                if ancilla > len(qc_m.qregs) or ancilla < 1: raise ValueError(" 'ancilla' should be less than no. of control registers and greater than 0")

                creg_cntrl = ClassicalRegister(len(qc_m.qregs[ancilla]))
                qc_m.add_register(creg_cntrl)
                qc_m.measure(qc_m.qregs[ancilla], creg_cntrl )

        elif isinstance(ancilla_specifier, str) and ancilla_specifier== "all":
            print('str')##cflag
            for reg in qc_m.qregs[1:] :
                creg = ClassicalRegister(len(reg))
                qc_m.add_register(creg)
                qc_m.measure(reg, creg)
                
    # plt.figure()
    counts = execute(qc_m, qsm, shots= shots).result().get_counts()
   
    if decimal_count_keys:
        counts_m = {}
        for key in counts.keys():
            split_key = key.split(' ')
            key_m = ''
            for string in split_key[:-1]:
                key_m+= str(int(string, 2)) + ' '
            key_m += split_key[-1][::-1] ##to-check
            counts_m[key_m] = counts[key]
        counts = counts_m

    if show_counts== True: print(counts)
    
    if return_counts: return counts
    else: return plot_histogram(counts)
    

   

####################################################################################################
                                ## permutation sub-routines ##
####################################################################################################



def bit_conditional( num: int, qc: QuantumCircuit, register: QuantumRegister): ## @helper_function
    """ helper function to assist the conditioning of ancillas 
    
        ARGS:
        ----
            num : the integer to be conditioned upon 
            qc : QuantumCircuit of the original circuit
            register : QuantumRegister corresponding to the ancilla which is conditioned upon 
        
        RETURNS:
        -------
            Works inplace i.e modifies the original circuit 'qc' that was passed """

    n = len(register)
    bin_str = format(num, "b").zfill(n)
    assert len(bin_str) == len(register)            ## need to assurethe bit ordering covention according tto qiskit's

    bin_str = bin_str[::-1]                         ## reverse the bit string
    for index, bit in enumerate(bin_str):
        if bit== '0': qc.x(register[index]) 
        
        

def generate_permutation_operators(permutation_index:int , power:int= 1)-> QuantumCircuit :
    """ Function to generate the permutaion operators 
    
        ARGS:
        ----
            permutation_index : index of the permutation operator 
            power : power of the permuation operator
            
        RETURNS:
        -------
            `QuantumCircuit` objects implementing the permutation operator
    """

    if permutation_index < 2 : raise ValueError(" 'permutation_index' must be greater than or equal to 2 ") 
    if power not in list(range(permutation_index)) : raise ValueError(" 'power' must be such that ; 0 <= 'power' < 'permutation_index'  ")

    qreg = QuantumRegister(permutation_index, name= 'qreg')
    qc = QuantumCircuit(qreg)

    for rep in range(power):    
        for qubit in range(permutation_index-1):
            qc.swap(qubit, qubit+1)

    return qc.to_gate(label= 'P_'+str(permutation_index)+'('+str(power)+')')




def append_permutation_operator(permutation_operator:int, power:int, qc:QuantumCircuit, qreg:QuantumRegister, control:QuantumRegister):
    """ Function to append a given permutation operator conditioned appropriately on the control ancillas on the quantum circuit

        ARGS:
        ----
            permutation_operator : index of the permutation operator 
            power : power of the permuation operator
            qc: `QuantumCircuit` to which the operator will be appended
            qreg: `QuantumRegister`storing the data 
            control: `QuantumRegister` upon which the action of the permutation operator is conditioned.
        
        RETURNS:
        -------
                `QuantumCircuit` 
    
    """
    
    permutation_size = permutation_operator.num_qubits 
    bit_conditional(power, qc, control)
    qc.append(permutation_operator.control(len(control)), [ control[i] for i in range(len(control))  ] + [ qreg[i] for i in range(permutation_size) ] )
    bit_conditional(power, qc, control)







##################################################################################################
                                 ## grover search sub-routines ##
##################################################################################################

## initial state preparation ~
def s_psi0(p):
    """ Prepare a QuantumCircuit that intiates a state required
        input:
            p= amplitude 
        output:
            s_psi0 gate    """
            
    qc = QuantumCircuit(1, name= " S_psi0 ")
    theta = 2*np.arcsin(np.sqrt(p))
    qc.ry(theta, 0)

    return qc.to_gate() 



## string to oracle ~
def str_to_oracle(pattern: str, name= 'oracle', return_type = "QuantumCircuit" ) -> Union[QuantumCircuit,  Statevector] :
    """ Convert a given string to an oracle circuit
        ARGS:
        ----
             pattern: a numpy vector with binarry entries 
             return_type: ['QuantumCircuit', 'Statevector']

        RETURNS:
        ------- 
               'QuantumCircuit' implementing the oracle
               'Statevector' evolved under the action of the oracle   """


    l = len(pattern)
    qr = QuantumRegister(l, name='reg')
    a = AncillaRegister(1, name='ancilla')
    oracle_circuit = QuantumCircuit(qr, a, name= name+'_'+ pattern )
    for q in range(l):
        if(pattern[q]=='0'): oracle_circuit.x(qr[q])
    oracle_circuit.x(a)
    oracle_circuit.h(a)
    oracle_circuit.mcx(qr, a)
    oracle_circuit.h(a)
    oracle_circuit.x(a)
    for q in range(l):
        if(pattern[q]=='0'): oracle_circuit.x(qr[q])
    
    #oracle_circuit.barrier()
    if return_type == "QuantumCircuit":
        return oracle_circuit
    elif return_type == "Statevector":
        return Statevector.from_instruction(oracle_circuit)



## oracle prep ~
def generate_oracles(good_states: list) -> QuantumCircuit :
    """ Return a QuantumCircuit that implements the oracles given the good_states
        ARGS:
        ----
            good_states: list of good staes, states must be binary strings, Eg. ['00', '11']

        RETURNS:
        -------
           ' QuantumCircuit' iplementing the oracle circuits """

    oracles = [ str_to_oracle(good_state) for good_state in good_states ]
    oracle_circuit = oracles[0]
    for oracle in oracles[1:] :
        oracle_circuit.compose(oracle,  inplace= True)
    

    return oracle_circuit

## diffuser prep ~
def diffuser(l:int)-> QuantumCircuit :
    """ Generate the Diffuser operator for the case where the initial state  is 
        the equal superposition state of all basis vectors 

        ARGS:
        ----
            l: no. of data qubits
        
        RETURNS:
        -------
                QuantumCircuit

    """

    qr = QuantumRegister(l, name='reg')
    a = AncillaRegister(1, name='ancilla')
    circuit = QuantumCircuit(qr, a, name= 'Diff.')
    
    circuit.h(qr)
    circuit.x(qr)
    
    circuit.x(a)
    circuit.h(a)
    circuit.mcx(qr ,a)
    circuit.h(a)
    circuit.x(a)

    circuit.x(qr)
    circuit.h(qr)
          
    return circuit

## grover prep ~
def grover(good_states: list, insert_barrier:bool= False)-> QuantumCircuit:
    
    oracle_circuit = generate_oracles(good_states)

    num_qubits= oracle_circuit.num_qubits - oracle_circuit.num_ancillas
    diffuser_circuit = diffuser(num_qubits)
    if insert_barrier== True: oracle_circuit.barrier()
    oracle_circuit.compose(diffuser_circuit, inplace= True)

    return oracle_circuit



