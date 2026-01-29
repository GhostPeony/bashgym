# Model Profile Implementation Plan

## Phase 1: Model Registry & Profile Foundation

### Task 1.1: Create ModelProfile dataclass
**File:** `bashgym/models/profile.py`
- Define ModelProfile with all fields from design
- Add serialization (to_dict, from_dict)
- Add save/load methods for model_profile.json

### Task 1.2: Create ModelRegistry class
**File:** `bashgym/models/registry.py`
- Scan data/models/ for existing runs
- Build index from trainer_state.json and adapter_config.json
- Create profiles for existing models
- Provide list/get/update/delete operations

### Task 1.3: Create API routes
**File:** `bashgym/api/models_routes.py`
- GET /api/models - list with filters
- GET /api/models/{model_id} - full profile
- POST /api/models/{model_id} - update metadata
- DELETE /api/models/{model_id} - archive
- POST /api/models/{model_id}/star - pin

### Task 1.4: Hook trainer to save profiles
**File:** `bashgym/gym/trainer.py` (modify)
- Save full TrainerConfig to profile on start
- Record loss curve during training
- Save final profile on completion

### Task 1.5: Create API types for frontend
**File:** `frontend/src/services/api.ts` (modify)
- Add ModelProfile interface
- Add modelsApi service object

### Task 1.6: Create ModelBrowser component
**File:** `frontend/src/components/models/ModelBrowser.tsx`
- Grid view with ModelCard components
- Filtering and sorting
- Navigation to profile

### Task 1.7: Create ModelCard component
**File:** `frontend/src/components/models/ModelCard.tsx`
- Display quick stats
- Star/pin functionality
- Compare/view actions

### Task 1.8: Create ModelProfile page
**File:** `frontend/src/components/models/ModelProfile.tsx`
- Header with identity
- Quick stats row
- Collapsible sections (stubbed for now)

### Task 1.9: Add routing
**File:** `frontend/src/App.tsx` (modify)
- Add /models route
- Add /models/:modelId route
