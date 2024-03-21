from dataclasses import dataclass, field
import .dag_generator as dag_generator
import .dag_dims_generator as dag_dims_generator
from .pick_weight import PickWeight
from typing import List, Set
import random

@dataclass
class OpNameGenRequirement:
    unary_elementwise_op_names: List[str] = field(
        default_factory=lambda: ["add"]
    )
    binary_elementwise_op_names: List[str] = field(
        default_factory=lambda: ["negative"]
    )
    broadcast_op_names: List[str] = field(
        default_factory=lambda: ["expand"]
    )
    reduce_op_names: List[str] = field(
        default_factory=lambda: ["reduce_sum"]
    )
    reshape_op_names: List[str] = field(
        default_factory=lambda: ["reshape"]
    )
    permute_op_names: List[str] = field(
        default_factory=lambda: ["transpose"]
    )
    source_op_names: List[str] = field(
        default_factory=lambda: ["zeros", "ones"]
    )

@dataclass
class Nope:
    @classmethod
    def MakeRandomInstance(
        cls,
        requirement: OpNameGenRequirement,
        dag_dims_gen_instruction: dag_dims_generator.DAGDimsGenInstruction,
        infer_ctx: dag_dims_generator.DimsIsEqOneInferContext
    ):
        return Nope()


@dataclass
class AddSinkTensor:

    @classmethod
    def MakeRandomInstance(
        cls,
        requirement: OpNameGenRequirement,
        dag_dims_gen_instruction: dag_dims_generator.DAGDimsGenInstruction,
        infer_ctx: dag_dims_generator.DimsIsEqOneInferContext
    ):
        return AddSinkTensor()

@dataclass
class AddUnaryOp:
    op_name: str

    @classmethod
    def MakeRandomInstance(
        cls,
        requirement: OpNameGenRequirement,
        dag_dims_gen_instruction: dag_dims_generator.DAGDimsGenInstruction,
        infer_ctx: dag_dims_generator.DimsIsEqOneInferContext
    ):
        input_dims_eq_one = dag_dims_gen_instruction.dims.source_tensor_dim_eq_one
        output_idx = dag_dims_gen_instruction.dag.source_tensor_index
        output_dims_eq_one = infer_ctx.current_source_tensor_dim_eq_one[output_idx]
        if _IsReduceOp(input_dims_eq_one, output_dims_eq_one):
            return AddUnaryOp(
                op_name=_GetRandomReduceOpName(requirement)
            )
        if _IsBroadcastOp(input_dims_eq_one, output_dims_eq_one):
            return AddUnaryOp(
                op_name=_GetRandomBroadcastOpName(requirement)
            )
        return AddUnaryOp(
            op_name=_GetRandomUnaryOpName(requirement)
        )


@dataclass
class AddBinaryOp:
    op_name: str

    @classmethod
    def MakeRandomInstance(
        cls,
        requirement: OpNameGenRequirement,
        dag_dims_gen_instruction: dag_dims_generator.DAGDimsGenInstruction,
        infer_ctx: dag_dims_generator.DimsIsEqOneInferContext
    ):
        return AddBinaryOp(
            op_name=_GetRandomBinaryOpName(requirement)
        )


@dataclass
class InsertBinaryOp:
    op_name: str

    @classmethod
    def MakeRandomInstance(
        cls,
        requirement: OpNameGenRequirement,
        dag_dims_gen_instruction: dag_dims_generator.DAGDimsGenInstruction,
        infer_ctx: dag_dims_generator.DimsIsEqOneInferContext
    ):
        return InsertBinaryOp(
            op_name=_GetRandomBinaryOpName(requirement)
        )


@dataclass
class AddBinaryClone:

    @classmethod
    def MakeRandomInstance(
        cls,
        requirement: OpNameGenRequirement,
        dag_dims_gen_instruction: dag_dims_generator.DAGDimsGenInstruction,
        infer_ctx: dag_dims_generator.DimsIsEqOneInferContext
    ):
        return AddBinaryClone()


@dataclass
class AddSourceOp:
    op_name: str

    @classmethod
    def MakeRandomInstance(
        cls,
        requirement: OpNameGenRequirement,
        dag_dims_gen_instruction: dag_dims_generator.DAGDimsGenInstruction,
        infer_ctx: dag_dims_generator.DimsIsEqOneInferContext
    ):
        return AddSourceOp(op_name=_GetRandomSourceOpName(requirement))


OpNameGenInstruction = ( Nope
                       | AddSinkTensor
                       | AddUnaryOp
                       | AddBinaryOp
                       | InsertBinaryOp
                       | AddBinaryClone
                       | AddSourceOp
                       )

kDAGGenClassToOpNameGenClassMap = {
    dag_generator.Nope: Nope,
    dag_generator.AddSinkTensor: AddSinkTensor,
    dag_generator.AddUnaryOp: AddUnaryOp,
    dag_generator.AddBinaryOp: AddBinaryOp,
    dag_generator.InsertBinaryOp: InsertBinaryOp,
    dag_generator.AddBinaryClone: AddBinaryClone,
    dag_generator.AddSourceOp: AddSourceOp,
}

def _IsReduceOp(input_dims_eq_one: List[bool], output_dims_eq_one:  List[bool]):
    return _IsLhsGreaterThanRhs(input_dims_eq_one, output_dims_eq_one)

def _IsBroadcastOp(input_dims_eq_one: List[bool], output_dims_eq_one:  List[bool]):
    return _IsLhsGreaterThanRhs(output_dims_eq_one, input_dims_eq_one)

def _IsLhsGreaterThanRhs(lhs: List[bool], rhs:  List[bool]):
    assert len(lhs) == len(rhs)
    for i in range(len(lhs)):
        if not (lhs[i] > rhs[i]):
            return False
    return True

def _GetRandomReduceOpName(requirement: DimGenRequirement):
    return _GetRandomOpName(requirement.reduce_op_names)

def _GetRandomBroadcastOpName(requirement: DimGenRequirement):
    return _GetRandomOpName(requirement.broadcast_op_names)

def _GetRandomUnaryOpName(requirement: DimGenRequirement):
    return _GetRandomOpName(requirement.unary_elementwise_op_names)

def _GetRandomBinaryOpName(requirement: DimGenRequirement):
    return _GetRandomOpName(requirement.binary_elementwise_op_names)

def _GetRandomSourceOpName(requirement: DimGenRequirement):
    return _GetRandomOpName(requirement.source_op_names)

def _GetRandomOpName(op_names: List[str]):
    kRangeMax = len(op_names) - 1
    assert 0 <= kRangeMax
    random_int = random.randomint(0, kRangeMax)
    return op_names[random_int]

class OpNameGenerator:
    def __init__(self, requirement: DimGenRequirement):
        self.requirement = requirement
    
    # Instructions generating sink nodes of DAG are on put the front of list.
    def Generate(
        self,
        dag_dims_gen_instructions: List[dag_dims_generator.DAGDimsGenInstruction]
    ) -> List[OpNameGenInstruction]:
        infer_ctx = dag_dims_generator.DimsIsEqOneInferContext()
        def CreateOpNameGenInstruction(dag_dims_gen_instruction):
            dag_gen_class = type(dag_dims_gen_instruction.dag)
            op_gen_class = kDAGGenClassToOpNameGenClassMap[dag_gen_class]
            op_name_gen_instruction = op_gen_class.MakeRandomInstance(
                self.requirement,
                dag_dims_gen_instruction,
                infer_ctx
            )
            infer_ctx.InferInputsDimsEqOne(dag_dims_gen_instruction)
            return op_name_gen_instruction
        return [
            CreateOpNameGenInstruction(dag_dims_gen_instruction)
            for dag_dims_gen_instruction in dag_dims_gen_instructions
        ]
