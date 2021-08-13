# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mlops/model_analysis/proto/config.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='mlops/model_analysis/proto/config.proto',
  package='mlops_model_analysis',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\'mlops/model_analysis/proto/config.proto\x12\x14mlops_model_analysis\"w\n\tModelSpec\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\x11\n\tmodel_ver\x18\x03 \x01(\t\x12\x12\n\nlabel_keys\x18\x04 \x03(\t\x12\x17\n\x0fprediction_keys\x18\x05 \x03(\t\x12\x1c\n\x14prediction_prob_keys\x18\x06 \x03(\t\"3\n\x0cMetricConfig\x12\x13\n\x0bmetric_name\x18\x01 \x01(\t\x12\x0e\n\x06\x63onfig\x18\x02 \x01(\t\"A\n\nMetricSpec\x12\x33\n\x07metrics\x18\x01 \x03(\x0b\x32\".mlops_model_analysis.MetricConfig\"^\n\nModelScore\x12\x12\n\nscore_name\x18\x01 \x01(\t\x12\x15\n\rreport_column\x18\x02 \x01(\t\x12\x12\n\nreport_row\x18\x03 \x01(\t\x12\x11\n\tthreshold\x18\x04 \x01(\x02\"\xaf\x01\n\nEvalConfig\x12\x33\n\nmodel_spec\x18\x02 \x01(\x0b\x32\x1f.mlops_model_analysis.ModelSpec\x12\x35\n\x0bmetric_spec\x18\x03 \x01(\x0b\x32 .mlops_model_analysis.MetricSpec\x12\x35\n\x0bmodel_score\x18\x04 \x01(\x0b\x32 .mlops_model_analysis.ModelScoreb\x06proto3'
)




_MODELSPEC = _descriptor.Descriptor(
  name='ModelSpec',
  full_name='mlops_model_analysis.ModelSpec',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='mlops_model_analysis.ModelSpec.name', index=0,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='model_ver', full_name='mlops_model_analysis.ModelSpec.model_ver', index=1,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='label_keys', full_name='mlops_model_analysis.ModelSpec.label_keys', index=2,
      number=4, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='prediction_keys', full_name='mlops_model_analysis.ModelSpec.prediction_keys', index=3,
      number=5, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='prediction_prob_keys', full_name='mlops_model_analysis.ModelSpec.prediction_prob_keys', index=4,
      number=6, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=65,
  serialized_end=184,
)


_METRICCONFIG = _descriptor.Descriptor(
  name='MetricConfig',
  full_name='mlops_model_analysis.MetricConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='metric_name', full_name='mlops_model_analysis.MetricConfig.metric_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='config', full_name='mlops_model_analysis.MetricConfig.config', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=186,
  serialized_end=237,
)


_METRICSPEC = _descriptor.Descriptor(
  name='MetricSpec',
  full_name='mlops_model_analysis.MetricSpec',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='metrics', full_name='mlops_model_analysis.MetricSpec.metrics', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=239,
  serialized_end=304,
)


_MODELSCORE = _descriptor.Descriptor(
  name='ModelScore',
  full_name='mlops_model_analysis.ModelScore',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='score_name', full_name='mlops_model_analysis.ModelScore.score_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='report_column', full_name='mlops_model_analysis.ModelScore.report_column', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='report_row', full_name='mlops_model_analysis.ModelScore.report_row', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='threshold', full_name='mlops_model_analysis.ModelScore.threshold', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=306,
  serialized_end=400,
)


_EVALCONFIG = _descriptor.Descriptor(
  name='EvalConfig',
  full_name='mlops_model_analysis.EvalConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='model_spec', full_name='mlops_model_analysis.EvalConfig.model_spec', index=0,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='metric_spec', full_name='mlops_model_analysis.EvalConfig.metric_spec', index=1,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='model_score', full_name='mlops_model_analysis.EvalConfig.model_score', index=2,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=403,
  serialized_end=578,
)

_METRICSPEC.fields_by_name['metrics'].message_type = _METRICCONFIG
_EVALCONFIG.fields_by_name['model_spec'].message_type = _MODELSPEC
_EVALCONFIG.fields_by_name['metric_spec'].message_type = _METRICSPEC
_EVALCONFIG.fields_by_name['model_score'].message_type = _MODELSCORE
DESCRIPTOR.message_types_by_name['ModelSpec'] = _MODELSPEC
DESCRIPTOR.message_types_by_name['MetricConfig'] = _METRICCONFIG
DESCRIPTOR.message_types_by_name['MetricSpec'] = _METRICSPEC
DESCRIPTOR.message_types_by_name['ModelScore'] = _MODELSCORE
DESCRIPTOR.message_types_by_name['EvalConfig'] = _EVALCONFIG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ModelSpec = _reflection.GeneratedProtocolMessageType('ModelSpec', (_message.Message,), {
  'DESCRIPTOR' : _MODELSPEC,
  '__module__' : 'mlops.model_analysis.proto.config_pb2'
  # @@protoc_insertion_point(class_scope:mlops_model_analysis.ModelSpec)
  })
_sym_db.RegisterMessage(ModelSpec)

MetricConfig = _reflection.GeneratedProtocolMessageType('MetricConfig', (_message.Message,), {
  'DESCRIPTOR' : _METRICCONFIG,
  '__module__' : 'mlops.model_analysis.proto.config_pb2'
  # @@protoc_insertion_point(class_scope:mlops_model_analysis.MetricConfig)
  })
_sym_db.RegisterMessage(MetricConfig)

MetricSpec = _reflection.GeneratedProtocolMessageType('MetricSpec', (_message.Message,), {
  'DESCRIPTOR' : _METRICSPEC,
  '__module__' : 'mlops.model_analysis.proto.config_pb2'
  # @@protoc_insertion_point(class_scope:mlops_model_analysis.MetricSpec)
  })
_sym_db.RegisterMessage(MetricSpec)

ModelScore = _reflection.GeneratedProtocolMessageType('ModelScore', (_message.Message,), {
  'DESCRIPTOR' : _MODELSCORE,
  '__module__' : 'mlops.model_analysis.proto.config_pb2'
  # @@protoc_insertion_point(class_scope:mlops_model_analysis.ModelScore)
  })
_sym_db.RegisterMessage(ModelScore)

EvalConfig = _reflection.GeneratedProtocolMessageType('EvalConfig', (_message.Message,), {
  'DESCRIPTOR' : _EVALCONFIG,
  '__module__' : 'mlops.model_analysis.proto.config_pb2'
  # @@protoc_insertion_point(class_scope:mlops_model_analysis.EvalConfig)
  })
_sym_db.RegisterMessage(EvalConfig)


# @@protoc_insertion_point(module_scope)