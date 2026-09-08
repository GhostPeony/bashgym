import { MainLayout } from './MainLayout'
import { SettingsModal } from '../common'
import { OnboardingModal } from '../onboarding/OnboardingModal'
import { useGlobalHotkeys } from '../../hooks/useHotkeys'

export function DesktopShell() {
  useGlobalHotkeys()
  return (
    <>
      <MainLayout />
      <SettingsModal />
      <OnboardingModal />
    </>
  )
}
