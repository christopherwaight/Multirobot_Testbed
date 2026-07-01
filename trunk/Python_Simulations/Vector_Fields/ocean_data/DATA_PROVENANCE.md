# HFR Ocean Data Provenance and Status

Data for reproducing Michini et al. (2014) Fig. 10/11.
Santa Barbara Channel, May 16 08:00 GMT through May 17 12:00 GMT 2012 (29 hourly files).
Source: Scripps/CORDC HFRNet USWC RTV product, NCEI Accession gov.noaa.nodc:IOOS-HFRadarRTVector.

## Inventory (verified 2026-07-01)

| Folder | Resolution | Files | SBC valid coverage | Status |
|--------|-----------|-------|--------------------|--------|
| `hfr_uswc_2012may/` | 6km | 29/29 | ~67% | Complete, usable |
| `hfr_uswc_2012may_2km/` | 2km | 29/29 | ~48-58% | Complete |
| `hfr_uswc_2012may_1km/` | 1km | 29/29 | ~0% | Not usable for SBC region |

## Provenance

All three resolutions were batch re-exported from CORDC's internal archive on 2020-01-21
by user `motero` using the tool `rtvUpdateNetcdf` (visible in each file's NC `history` attribute).

All files (original 19 and the recovered 10) came from NCEI's THREDDS-Ocean server:
  https://www.ncei.noaa.gov/thredds-ocean/fileServer/ioos/hfradar/rtv/2012/201205/USWC/

The UCSD/Scripps server (hfrnet-tds.ucsd.edu) was decommissioned June 30, 2025, over a
year before these files were downloaded (June 2026). UCSD was never the source.

The original 19 files were downloaded in a prior session (June 25, 2026). The download
stopped after 19 files -- likely a scripted loop or transient interruption. The NCEI server
has all 29 hours and the `history` timestamps confirm the CORDC export job ran sequentially
through all 29 (23:06:24 through 23:06:41 UTC on 2020-01-21), so all 29 were always
available on NCEI. The gap was in the download, not the archive.

## 2km gap resolved (2026-07-01)

The 10 missing 2km hours (170300-171200) were downloaded from NCEI's THREDDS-Ocean server:

  https://www.ncei.noaa.gov/thredds-ocean/fileServer/ioos/hfradar/rtv/2012/201205/USWC/

All 10 files confirmed valid: 48-56% SBC coverage, consistent with the first 19 hours.
The 2km dataset is now complete (29/29 hours).

This is also the source for all previously downloaded files (confirmed from session transcript
dated 2026-06-25). The path structure is:
  .../ioos/hfradar/rtv/{year}/{yearmonth}/USWC/{timestamp}_hfr_uswc_{res}_rtv_uwls_SIO.nc

## Dead ends (do not retry)

- hfrnet-tds.ucsd.edu: permanently decommissioned June 30, 2025
- dods.ndbc.noaa.gov/thredds: rolling recent-window only, no 2012 data
- coastwatch.pfeg.noaa.gov/erddap (ucsdHfrW2, ucsdHfrW6): rolling ~90-day cache only
- Zenodo, Figshare, GitHub: no copies found
- NCEI OAI-PMH and THREDDS endpoints: 404 for all tested paths
