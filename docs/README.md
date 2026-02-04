# Documentation Index

Welcome to the GPU Cluster Acceptance Testing Tool documentation.

## 🚀 Quick Start

**New to this tool?** Start here:
1. Read the main [README.md](../README.md)
2. Read [HOW_IT_WORKS.md](HOW_IT_WORKS.md) to understand the architecture
3. Review [QUICK_SUMMARY.md](QUICK_SUMMARY.md) for recent improvements
4. Try the examples in `../examples/`

## 📚 Documentation Structure

### Core Documentation

#### [HOW_IT_WORKS.md](HOW_IT_WORKS.md) ⭐⭐⭐
**Complete explanation of tool architecture and methodology**

What you'll find:
- How data is generated (NO DATABASE - synthetic)
- Container image structure
- Complete execution flow diagrams
- What gets tested (GPU compute + NCCL communication)
- Multi-node communication patterns
- Why synthetic data works for cluster validation

**Start here to understand the tool.**

#### [COMPLETE_SUMMARY.md](COMPLETE_SUMMARY.md) ⭐
**Complete overview of all enhancements made to the tool**

What you'll find:
- All new capabilities added
- Complete file inventory
- Performance benchmarks
- Quick start examples
- Troubleshooting guide

**Use this for a comprehensive feature overview.**

---

### Testing Guides

#### [NCCL_TESTING.md](NCCL_TESTING.md)
**Complete guide to NCCL bandwidth and latency testing**

Topics covered:
- When to use NCCL tests vs full training tests
- Running NCCL tests on Kubernetes, Slurm, bare metal
- Testing specific network modes (NVLink, InfiniBand, Ethernet)
- Interpreting NCCL test output
- Troubleshooting low bandwidth issues
- Reference benchmarks for H100/A100

**Use this when**: You need to validate network performance or debug InfiniBand issues.

#### [TESTING_WORKFLOW.md](TESTING_WORKFLOW.md)
**Visual decision trees and workflow diagrams**

Contains:
- Decision tree: Which test to run?
- Recommended testing sequence
- Troubleshooting decision tree
- Test comparison matrix
- Expected results timeline

**Use this when**: You need to decide which tests to run or understand the overall testing strategy.

#### [ACCEPTANCE_PLAYBOOK.md](ACCEPTANCE_PLAYBOOK.md)
**Benchmarks and troubleshooting for infrastructure engineers**

Includes:
- H100 and A100 performance baselines
- Detailed NCCL debugging steps
- Acceptance criteria for different hardware
- Common issues and solutions
- Performance tuning tips

**Use this when**: You're performing acceptance testing on new clusters or troubleshooting performance issues.

---

### Nebius Production Patterns

#### [LEARNINGS_FROM_NEBIUS.md](LEARNINGS_FROM_NEBIUS.md) ⭐⭐
**Analysis of Nebius production tests and best practices**

What you'll find:
- Key findings from Nebius NCCL and Ray tests
- Resource configuration patterns (multi-GPU, memory, CPU)
- Shared memory configuration (critical!)
- Init container patterns (ulimit)
- NCCL environment variables
- Comparison matrix: Nebius vs our tool
- Implementation priorities

**Use this when**: Deploying to Nebius clusters or learning production patterns.

#### [INFINIBAND_CONFIGURATION.md](INFINIBAND_CONFIGURATION.md) ⭐⭐
**Complete guide to NCCL and InfiniBand configuration**

Topics covered:
- Network topology patterns (NVLink, InfiniBand, RoCE, Cloud)
- NCCL environment variables reference
- Nebius-specific H100 configuration
- Diagnostic commands
- Performance troubleshooting
- Validation checklist

**Use this when**: Configuring NCCL for InfiniBand/RDMA clusters or debugging network performance.

#### [IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md)
**Complete improvements based on Nebius tests**

Contains:
- Executive summary of changes
- Critical improvements implemented
- New files created
- Comparison matrix (before/after)
- Validation results
- Impact analysis

**Use this when**: You need to understand what changed and why.

#### [QUICK_SUMMARY.md](QUICK_SUMMARY.md)
**One-page summary of improvements**

Quick reference:
- Critical gaps found
- What we implemented
- Quick comparison table
- How to use
- Key advantages maintained

**Use this when**: You need a quick overview of recent improvements.

---

### Implementation Details

#### [CLEANUP_GUIDE.md](CLEANUP_GUIDE.md)
**Resource cleanup and lifecycle management**

Topics covered:
- Cleanup behavior by platform (Docker, Kubernetes, Slurm)
- Automatic cleanup configuration (TTL, policies)
- Manual cleanup commands
- Automated cleanup scripts
- Storage considerations and best practices
- Monitoring cleanup effectiveness

**Use this when**: You need to manage test resource cleanup or prevent accumulation of completed jobs.

#### [Exercise 2 Summary.md](Exercise%202%20Summary.md)
**Detailed analysis of Nebius Soperator test patterns**

Contains:
- Complete Soperator test structure analysis
- Comparison: Soperator vs our tool
- All improvements made based on learnings
- Implementation statistics
- Key learnings and takeaways

**Use this when**: You want to understand the design decisions behind the NCCL testing enhancements.

#### [IMPROVEMENTS_FROM_SOPERATOR.md](IMPROVEMENTS_FROM_SOPERATOR.md)
**Quick reference for improvements made**

Provides:
- TL;DR of what changed
- New capabilities summary
- Soperator patterns adopted (and not adopted)
- Quick troubleshooting guide
- One-page reference

**Use this when**: You need a quick overview of the enhancements without deep details.

#### [Exercise 2 Implementation Plan.md](Exercise%202%20Implementation%20Plan.md)
**Original requirements and implementation checklist**

Documents:
- Original functional requirements
- Task breakdown (all completed ✅)
- Technical specifications
- Implementation status

**Use this when**: You want to see the original requirements or verify completeness.

---

### Infrastructure

#### [NEBIUS_REGISTRY_GUIDE.md](NEBIUS_REGISTRY_GUIDE.md)
**Complete guide to Nebius Container Registry**

Topics:
- Setting up Nebius CLI authentication
- Building and pushing containers
- Common registry errors and solutions
- CI/CD integration examples
- Best practices

**Use this when**: You need to build, push, or pull images from Nebius Container Registry.

---

## 🎯 Common Tasks → Documentation Mapping

### "I want to validate a new GPU cluster"
→ Start with [TESTING_WORKFLOW.md](TESTING_WORKFLOW.md)  
→ Then follow steps in [NCCL_TESTING.md](NCCL_TESTING.md) + main README

### "The network seems slow"
→ Read [NCCL_TESTING.md](NCCL_TESTING.md) - Section: "Troubleshooting"  
→ Then check [ACCEPTANCE_PLAYBOOK.md](ACCEPTANCE_PLAYBOOK.md) - "NCCL debugging"

### "How do I clean up test resources?"
→ See [CLEANUP_GUIDE.md](CLEANUP_GUIDE.md)  
→ Or check main README - "Resource Cleanup" section

### "I need to understand what changed in the tool"
→ Read [COMPLETE_SUMMARY.md](COMPLETE_SUMMARY.md)  
→ Or [IMPROVEMENTS_FROM_SOPERATOR.md](IMPROVEMENTS_FROM_SOPERATOR.md) for quick overview

### "Should I use NCCL tests or training tests?"
→ See [TESTING_WORKFLOW.md](TESTING_WORKFLOW.md) - "Decision Tree"  
→ Or [NCCL_TESTING.md](NCCL_TESTING.md) - "When to Use Each Approach"

### "What performance should I expect?"
→ See [ACCEPTANCE_PLAYBOOK.md](ACCEPTANCE_PLAYBOOK.md) - "Expected Performance"  
→ Or [COMPLETE_SUMMARY.md](COMPLETE_SUMMARY.md) - "Performance Expectations"

### "How do I work with Nebius Container Registry?"
→ Read [NEBIUS_REGISTRY_GUIDE.md](NEBIUS_REGISTRY_GUIDE.md)

### "Why was it designed this way?"
→ Read [Exercise 2 Summary.md](Exercise%202%20Summary.md)

---

## 📊 Documentation Statistics

| Document | Pages | Primary Audience | Time to Read |
|----------|-------|------------------|--------------|
| COMPLETE_SUMMARY.md | 12 | Everyone | 15 min |
| NCCL_TESTING.md | 18 | Infrastructure Engineers | 25 min |
| TESTING_WORKFLOW.md | 8 | Test Operators | 10 min |
| ACCEPTANCE_PLAYBOOK.md | 22 | Infrastructure Engineers | 30 min |
| Exercise 2 Summary.md | 20 | Developers, Architects | 30 min |
| IMPROVEMENTS_FROM_SOPERATOR.md | 10 | Quick Reference | 8 min |
| NEBIUS_REGISTRY_GUIDE.md | 12 | DevOps, CI/CD | 15 min |
| Exercise 2 Implementation Plan.md | 3 | Project Management | 5 min |

**Total**: ~2,100 lines of comprehensive documentation

---

## 🔍 Search Tips

### By Keyword

- **Performance benchmarks**: See ACCEPTANCE_PLAYBOOK.md or COMPLETE_SUMMARY.md
- **NCCL configuration**: See NCCL_TESTING.md
- **InfiniBand troubleshooting**: See NCCL_TESTING.md or ACCEPTANCE_PLAYBOOK.md
- **Container registry**: See NEBIUS_REGISTRY_GUIDE.md
- **Kubernetes examples**: See main README.md
- **Slurm examples**: See main README.md or NCCL_TESTING.md
- **Network debugging**: See NCCL_TESTING.md or ACCEPTANCE_PLAYBOOK.md
- **Design decisions**: See Exercise 2 Summary.md

### By Use Case

- **Quick validation** (< 5 min): NCCL_TESTING.md
- **Full acceptance** (< 30 min): ACCEPTANCE_PLAYBOOK.md + main README
- **Debugging slow training**: TESTING_WORKFLOW.md → NCCL_TESTING.md
- **Learning the tool**: COMPLETE_SUMMARY.md → main README
- **CI/CD setup**: NEBIUS_REGISTRY_GUIDE.md

---

## 🤝 Contributing

When adding new documentation:

1. **Update this index** with the new document
2. **Cross-reference** related documents
3. **Add to task mapping** if applicable
4. **Keep consistent formatting** with existing docs
5. **Include visual aids** when helpful (diagrams, tables, code blocks)

---

## 📞 Support

### Documentation Issues

If you find errors or have suggestions:
1. Check if the information is in another document
2. Review the complete summary for context
3. Consult the testing workflow for procedural questions

### Tool Issues

For tool-related issues:
1. Check [ACCEPTANCE_PLAYBOOK.md](ACCEPTANCE_PLAYBOOK.md) troubleshooting section
2. Run verification script: `./scripts/verify-k8s-gpu-cluster.sh`
3. Enable NCCL debug: `NCCL_DEBUG=INFO`

---

## 📈 Version History

### Current Version (Post-Soperator Enhancement)

**Features**:
- ✅ Full training tests (ResNet-50, Transformer)
- ✅ NCCL bandwidth tests (all_reduce_perf)
- ✅ Kubernetes, Slurm, bare metal support
- ✅ Mixed GPU/non-GPU cluster support
- ✅ InfiniBand/RDMA detection
- ✅ Comprehensive documentation

**Documentation**:
- 8 comprehensive guides
- ~2,100 lines of documentation
- Visual workflows and decision trees
- Complete troubleshooting guides

### Original Version

**Features**:
- Full training tests only
- Kubernetes, Slurm, bare metal support
- Basic documentation

---

## 🎯 Next Steps

After reading the documentation:

1. **Try the tool**:
   ```bash
   docker pull cr.eu-north1.nebius.cloud/e00tnz9wpyxva2s992/gpu_cluster_testing:latest
   docker run --gpus all --rm cr.eu-north1.nebius.cloud/e00tnz9wpyxva2s992/gpu_cluster_testing:latest \
     --model resnet50 --batch-size 32
   ```

2. **Run NCCL tests**:
   ```bash
   sbatch examples/slurm-nccl-test.sh
   ```

3. **Join the discussion**: Share your results and learnings with the team

---

**Last Updated**: [Current Date]  
**Maintained By**: Nebius Infrastructure Engineering  
**Repository**: `/Users/ahb/gpu_cluster_testing`
